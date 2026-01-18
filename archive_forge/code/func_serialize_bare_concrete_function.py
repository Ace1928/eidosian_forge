from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def serialize_bare_concrete_function(concrete_function):
    """Build a SavedBareConcreteFunction."""
    proto = saved_object_graph_pb2.SavedBareConcreteFunction(concrete_function_name=concrete_function.name, allowed_positional_arguments=concrete_function._num_positional_args, argument_keywords=concrete_function._arg_keywords)
    function_spec = get_preinitialized_function_spec(concrete_function)
    if function_spec is not None:
        proto.function_spec.CopyFrom(_serialize_function_spec(function_spec))
    return proto