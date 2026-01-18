from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def serialize_function(function, concrete_functions):
    """Build a SavedFunction proto."""
    proto = saved_object_graph_pb2.SavedFunction()
    function_spec_proto = _serialize_function_spec(function.function_spec)
    proto.function_spec.CopyFrom(function_spec_proto)
    for concrete_function in concrete_functions:
        proto.concrete_functions.append(concrete_function.name)
    return proto