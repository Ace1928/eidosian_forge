from tensorflow.python.util.tf_export import tf_export
def get_get_session_function():
    global _KERAS_GET_SESSION_FUNCTION
    return _KERAS_GET_SESSION_FUNCTION