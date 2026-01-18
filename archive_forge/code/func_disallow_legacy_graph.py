from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
def disallow_legacy_graph(cls_name, method_name):
    if not ops.executing_eagerly_outside_functions():
        error_msg = 'Calling `{cls_name}.{method_name}` in graph mode is not supported when the `{cls_name}` instance was constructed with eager mode enabled. Please construct your `{cls_name}` instance in graph mode or call `{cls_name}.{method_name}` with eager mode enabled.'
        error_msg = error_msg.format(cls_name=cls_name, method_name=method_name)
        raise ValueError(error_msg)