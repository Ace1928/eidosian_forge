from tensorflow.python.util.tf_export import tf_export
def get_call_context_function():
    global _KERAS_CALL_CONTEXT_FUNCTION
    return _KERAS_CALL_CONTEXT_FUNCTION