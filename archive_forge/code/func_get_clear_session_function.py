from tensorflow.python.util.tf_export import tf_export
def get_clear_session_function():
    global _KERAS_CLEAR_SESSION_FUNCTION
    return _KERAS_CLEAR_SESSION_FUNCTION