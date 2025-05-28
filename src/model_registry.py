def register_model(model, model_name, n_estimators, accuracy):
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Mi primer Modelo")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

def display_model_info(model_name, n_estimators, accuracy):
    """
    Función simple para mostrar información del modelo.
    Esta función es solo informativa y no afecta la funcionalidad principal.
    """
    print(f"Información del Modelo:")
    print(f"Nombre: {model_name}")
    print(f"Número de estimadores: {n_estimators}")
    print(f"Precisión: {accuracy:.2%}")

def check_model_quality(accuracy, min_accuracy=0.7, min_estimators=50):
    """
    Verifica si el modelo cumple con criterios mínimos de calidad.
    
    Args:
        accuracy (float): Precisión del modelo
        min_accuracy (float): Precisión mínima aceptable (default: 0.7)
        min_estimators (int): Número mínimo de estimadores (default: 50)
    
    Returns:
        dict: Diccionario con el estado de cada criterio
    """
    quality_metrics = {
        "cumple_precision": accuracy >= min_accuracy,
        "precision_actual": accuracy,
        "precision_minima": min_accuracy
    }
    
    return quality_metrics