from tensorflow.python.util.tf_export import tf_export
def set_non_mixed_precision_session_created(created):
    global _non_mixed_precision_session_created
    _non_mixed_precision_session_created = created