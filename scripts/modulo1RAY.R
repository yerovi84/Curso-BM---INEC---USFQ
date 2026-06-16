paquetes <- c("tidymodels", "tidyverse", "broom", "glmnet", "readxl")

suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(broom)
  library(readxl)
})

tidymodels_prefer()
theme_set(theme_minimal(base_size = 12))

semilla_global <- 123
set.seed(semilla_global)

output_dir <- file.path("output", "modulo1RAY")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Cargar datos reales del INEC y preparar nombres sencillos para modelar.
data_inec_original <- suppressMessages(read_excel("data/data_pobreza_INEC.xlsx"))
data_inec <- data_inec_original
names(data_inec) <- make.names(names(data_inec), unique = TRUE)

datos_pobreza <- data_inec %>%
  rename(
    canton = Canton,
    id_canton = CodCanton...2,
    nbi_mef = NBIMEF
  ) %>%
  mutate(
    canton = as.character(canton),
    pobre = factor(
      if_else(nbi_mef >= median(nbi_mef, na.rm = TRUE), "Pobre", "No pobre"),
      levels = c("No pobre", "Pobre")
    )
  ) %>%
  select(-CodCanton...3)

datos_pobreza %>% glimpse()

resumen_pobreza <- datos_pobreza %>%
  summarise(
    n = n(),
    nbi_promedio = mean(nbi_mef, na.rm = TRUE),
    nbi_mediano = median(nbi_mef, na.rm = TRUE),
    tasa_pobreza = mean(pobre == "Pobre", na.rm = TRUE)
  )

missing_pobreza <- datos_pobreza %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  filter(n_missing > 0)

write_csv(resumen_pobreza, file.path(output_dir, "resumen_pobreza.csv"))
write_csv(missing_pobreza, file.path(output_dir, "missing_pobreza.csv"))

resumen_pobreza
missing_pobreza

# Función auxiliar para imputar categorías con la moda.
moda <- function(x) {
  valores <- x[!is.na(x)]
  if (length(valores) == 0) return(NA)
  valores[which.max(tabulate(match(valores, unique(valores))))]
}

datos_pobreza_imputados <- datos_pobreza %>%
  mutate(
    across(where(is.numeric), ~ replace_na(.x, mean(.x, na.rm = TRUE))),
    across(where(~ is.character(.x) || is.factor(.x)), ~ replace_na(.x, moda(.x)))
  )

write_csv(datos_pobreza_imputados, file.path(output_dir, "datos_pobreza_imputados_media_moda.csv"))

datos_pobreza_imputados %>% summarise(across(everything(), ~ sum(is.na(.x))))

variables_numericas_modelo <- datos_pobreza_imputados %>%
  select(where(is.numeric), -id_canton, -nbi_mef, -NBIenemdu, -nbicenso2022) %>%
  names()

parametros_normalizacion <- datos_pobreza_imputados %>%
  summarise(across(
    all_of(variables_numericas_modelo),
    list(media = mean, desviacion = sd),
    .names = "{.col}_{.fn}"
  )) %>%
  pivot_longer(
    everything(),
    names_to = c("variable", "estadistico"),
    names_pattern = "(.+)_(media|desviacion)",
    values_to = "valor"
  ) %>%
  pivot_wider(names_from = estadistico, values_from = valor)

datos_pobreza_normalizados <- datos_pobreza_imputados %>%
  mutate(across(
    all_of(variables_numericas_modelo),
    ~ (.x - mean(.x)) / sd(.x),
    .names = "{.col}_z"
  ))

write_csv(parametros_normalizacion, file.path(output_dir, "parametros_normalizacion.csv"))
write_csv(datos_pobreza_normalizados, file.path(output_dir, "datos_pobreza_normalizados.csv"))

parametros_normalizacion
datos_pobreza_normalizados %>%
  select(canton, id_canton, nbi_mef, pobre, ends_with("_z")) %>%
  glimpse()

datos_pobreza %>%
  ggplot(aes(x = nbi_mef, fill = pobre)) +
  geom_histogram(bins = round(nrow(datos_pobreza)^(1 / 2), 0), alpha = .75, color = "white") +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "NBI MEF", y = "Cantones", fill = "Condición", title = "Distribución de pobreza por NBI")

prop_pobre_real <- mean(datos_pobreza$pobre == "Pobre")

set.seed(semilla_global)
split_simple <- initial_split(datos_pobreza, prop = 0.80)
train_simple <- training(split_simple)
test_simple <- testing(split_simple)

set.seed(semilla_global)
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

train <- training(split_pobreza)
test <- testing(split_pobreza)

set.seed(semilla_global)
folds_simple <- vfold_cv(datos_pobreza, v = 5)

set.seed(semilla_global)
folds_strata <- vfold_cv(datos_pobreza, v = 5, strata = pobre)

set.seed(semilla_global)
boots_simple <- bootstraps(datos_pobreza, times = 5)

set.seed(semilla_global)
boots_strata <- bootstraps(datos_pobreza, times = 5, strata = pobre)

resumen_particion <- tibble(
  particion = c(
    "Datos completos",
    "Initial split simple - Train",
    "Initial split simple - Test",
    "Initial split con strata - Train",
    "Initial split con strata - Test",
    "K-fold simple - Fold 1",
    "K-fold con strata - Fold 1",
    "Bootstrap simple - Muestra 1",
    "Bootstrap con strata - Muestra 1"
  ),
  prop_pobre = c(
    prop_pobre_real,
    mean(train_simple$pobre == "Pobre"),
    mean(test_simple$pobre == "Pobre"),
    mean(train$pobre == "Pobre"),
    mean(test$pobre == "Pobre"),
    mean(assessment(folds_simple$splits[[1]])$pobre == "Pobre"),
    mean(assessment(folds_strata$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_simple$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_strata$splits[[1]])$pobre == "Pobre")
  )
) %>%
  mutate(
    prop_real = prop_pobre_real,
    diferencia_vs_real = prop_pobre - prop_real
  )

write_csv(resumen_particion, file.path(output_dir, "resumen_particion.csv"))

resumen_particion

predictores_modelo <- c(
  "PorcentajeInstConInternet",
  "TEF11a19madre",
  "CajerosAutomáticosTasapobmay15años",
  "TotPuntosAteFinTasapobmay15años",
  "PIBpercap",
  "TasaGlobalFecundmadre",
  "ParticipacionPIBEnseñanza",
  "Porcentajenacprematuromoderadomujermujermadre",
  "aguaPotableViv",
  "OficinasTasapobmay15años"
)
predictores_modelo <- make.names(predictores_modelo, unique = TRUE)

set.seed(semilla_global)
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

train <- training(split_pobreza)
test <- testing(split_pobreza)

receta_ols <- recipe(
  reformulate(predictores_modelo, response = "nbi_mef"),
  data = train
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

modelo_ols <- linear_reg() %>%
  set_engine("lm")

wf_ols <- workflow() %>%
  add_recipe(receta_ols) %>%
  add_model(modelo_ols)

wf_ols

ajuste_ols <- fit(wf_ols, data = train)

pred_ols <- predict(ajuste_ols, new_data = test) %>%
  bind_cols(test %>% select(nbi_mef))

head(pred_ols)

metr <- metric_set(rmse, rsq, mae)

metricas_ols <- metr(
  pred_ols,
  truth = nbi_mef,
  estimate = .pred
)

write_csv(metricas_ols, file.path(output_dir, "metricas_ols.csv"))

set.seed(semilla_global)
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

train <- training(split_pobreza)
test <- testing(split_pobreza)

receta_logit <- recipe(
  reformulate(predictores_modelo, response = "pobre"),
  data = train
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

modelo_logit <- logistic_reg() %>%
  set_engine("glm")

wf_logit <- workflow() %>%
  add_recipe(receta_logit) %>%
  add_model(modelo_logit)

wf_logit

ajuste_logit <- fit(wf_logit, data = train)

pred_logit <- predict(ajuste_logit, new_data = test, type = "prob") %>%
  bind_cols(predict(ajuste_logit, new_data = test, type = "class")) %>%
  bind_cols(test %>% select(pobre))

head(pred_logit)

metr <- metric_set(
  accuracy,
  precision,
  recall,
  f_meas,
  roc_auc
)

metricas_logit <- metr(
  pred_logit,
  truth = pobre,
  estimate = .pred_class,
  .pred_Pobre,
  event_level = "second"
)

matriz_confusion <- conf_mat(
  pred_logit,
  truth = pobre,
  estimate = .pred_class
)

write_csv(metricas_logit, file.path(output_dir, "metricas_logit.csv"))
write_csv(tidy(matriz_confusion), file.path(output_dir, "matriz_confusion_logit.csv"))

metricas_logit
matriz_confusion

# Ejercicio final: predecir los cantones con mayor pobreza por NBI.
# NBIMEF es la variable objetivo continua; "pobre" identifica cantones con NBI
# igual o superior a la mediana observada.

pred_logit_train <- predict(ajuste_logit, new_data = train, type = "prob") %>%
  bind_cols(predict(ajuste_logit, new_data = train, type = "class")) %>%
  bind_cols(train %>% select(pobre))

matriz_confusion_train <- conf_mat(
  pred_logit_train,
  truth = pobre,
  estimate = .pred_class
)

matriz_confusion_test <- matriz_confusion

write_csv(tidy(matriz_confusion_train), file.path(output_dir, "matriz_confusion_logit_train.csv"))
write_csv(tidy(matriz_confusion_test), file.path(output_dir, "matriz_confusion_logit_test.csv"))

matriz_confusion_train
matriz_confusion_test

# Modelos finales entrenados con todos los cantones disponibles para producir
# el ranking operativo de pobreza cantonal.
receta_ols_final <- recipe(
  reformulate(predictores_modelo, response = "nbi_mef"),
  data = datos_pobreza
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

wf_ols_final <- workflow() %>%
  add_recipe(receta_ols_final) %>%
  add_model(modelo_ols)

ajuste_ols_final <- fit(wf_ols_final, data = datos_pobreza)

receta_logit_final <- recipe(
  reformulate(predictores_modelo, response = "pobre"),
  data = datos_pobreza
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

wf_logit_final <- workflow() %>%
  add_recipe(receta_logit_final) %>%
  add_model(modelo_logit)

ajuste_logit_final <- fit(wf_logit_final, data = datos_pobreza)

predicciones_cantones <- datos_pobreza %>%
  select(canton, id_canton, nbi_mef, pobre) %>%
  bind_cols(predict(ajuste_ols_final, new_data = datos_pobreza) %>% rename(nbi_mef_predicha = .pred)) %>%
  bind_cols(predict(ajuste_logit_final, new_data = datos_pobreza, type = "prob")) %>%
  bind_cols(predict(ajuste_logit_final, new_data = datos_pobreza, type = "class") %>% rename(pobre_predicho = .pred_class)) %>%
  mutate(
    error_nbi = nbi_mef - nbi_mef_predicha,
    ranking_pobreza_predicha = min_rank(desc(nbi_mef_predicha))
  ) %>%
  arrange(ranking_pobreza_predicha)

cantones_mas_pobres_predichos <- predicciones_cantones %>%
  slice_head(n = 20)

matriz_confusion_todos_cantones <- conf_mat(
  predicciones_cantones,
  truth = pobre,
  estimate = pobre_predicho
)

write_csv(predicciones_cantones, file.path(output_dir, "predicciones_todos_cantones.csv"))
write_csv(cantones_mas_pobres_predichos, file.path(output_dir, "cantones_mas_pobres_predichos.csv"))
write_csv(tidy(matriz_confusion_todos_cantones), file.path(output_dir, "matriz_confusion_logit_todos_cantones.csv"))

cantones_mas_pobres_predichos
matriz_confusion_todos_cantones
