from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def serialize_concrete_function(concrete_function, node_ids):
    """Build a SavedConcreteFunction."""
    bound_inputs = []
    try:
        for capture in concrete_function.captured_inputs:
            bound_inputs.append(node_ids[capture])
    except KeyError:
        raise KeyError(f"Failed to add concrete function '{concrete_function.name}' to object-based SavedModel as it captures tensor {capture!r} which is unsupported or not reachable from root. One reason could be that a stateful object or a variable that the function depends on is not assigned to an attribute of the serialized trackable object (see SaveTest.test_captures_unreachable_variable).")
    concrete_function_proto = saved_object_graph_pb2.SavedConcreteFunction()
    structured_outputs = func_graph_module.convert_structure_to_signature(concrete_function.structured_outputs)
    concrete_function_proto.canonicalized_input_signature.CopyFrom(nested_structure_coder.encode_structure(concrete_function.structured_input_signature))
    concrete_function_proto.output_signature.CopyFrom(nested_structure_coder.encode_structure(structured_outputs))
    concrete_function_proto.bound_inputs.extend(bound_inputs)
    return concrete_function_proto