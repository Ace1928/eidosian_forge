from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat
def get_handle_data(source_t):
    """Obtains HandleData from a tensor."""
    if isinstance(source_t, core.Value):
        return source_t._handle_data
    return get_resource_handle_data(source_t)