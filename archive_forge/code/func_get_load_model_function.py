from tensorflow.python.util.tf_export import tf_export
def get_load_model_function():
    global _KERAS_LOAD_MODEL_FUNCTION
    return _KERAS_LOAD_MODEL_FUNCTION