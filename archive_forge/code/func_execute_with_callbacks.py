from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def execute_with_callbacks(op_name, num_outputs, inputs, attrs, ctx, name=None):
    """Monkey-patch to execute to enable execution callbacks."""
    tensors = quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
    for callback in ctx.op_callbacks:
        callback(op_name, tuple(inputs), attrs, tensors, name)
    return tensors