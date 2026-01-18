from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat
def set_handle_data(target_t, handle_data):
    """Sets handle data on the giver tensor."""
    if handle_data is None or not handle_data.is_set or (not handle_data.shape_and_type):
        return
    if isinstance(target_t, core.Value):
        target_t._handle_data = handle_data
        return
    with target_t.graph._c_graph.get() as c_graph:
        pywrap_tf_session.SetHandleShapeAndType(c_graph, target_t._as_tf_output(), handle_data.SerializeToString())