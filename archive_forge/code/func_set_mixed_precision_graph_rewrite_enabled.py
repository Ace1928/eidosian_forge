from tensorflow.python.util.tf_export import tf_export
def set_mixed_precision_graph_rewrite_enabled(enabled):
    global _mixed_precision_graph_rewrite_is_enabled
    _mixed_precision_graph_rewrite_is_enabled = enabled