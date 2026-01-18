from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def make_tensor(v, arg_name):
    """Ensure v is a TensorProto."""
    if isinstance(v, tensor_pb2.TensorProto):
        return v
    elif isinstance(v, str):
        pb = tensor_pb2.TensorProto()
        text_format.Merge(v, pb)
        return pb
    raise TypeError("Don't know how to convert %s to a TensorProto for argument '%s'." % (repr(v), arg_name))