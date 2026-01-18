from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def make_shape(v, arg_name):
    """Convert v into a list."""
    try:
        shape = tensor_shape.as_shape(v)
    except TypeError as e:
        raise TypeError('Error converting %s to a TensorShape: %s.' % (arg_name, e))
    except ValueError as e:
        raise ValueError('Error converting %s to a TensorShape: %s.' % (arg_name, e))
    if shape.ndims is None:
        return None
    else:
        return shape.as_list()