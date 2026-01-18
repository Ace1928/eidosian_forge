from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat
def create_handle_data(shape, dtype):
    handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    handle_data.is_set = True
    handle_data.shape_and_type.append(cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(shape=shape.as_proto(), dtype=dtype.as_datatype_enum))
    return handle_data