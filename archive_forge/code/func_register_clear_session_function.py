from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.register_clear_session_function', v1=[])
def register_clear_session_function(func):
    global _KERAS_CLEAR_SESSION_FUNCTION
    _KERAS_CLEAR_SESSION_FUNCTION = func