from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def get_output_slot(element_name):
    """Get the output slot number from the name of a graph element.

  If element_name is a node name without output slot at the end, 0 will be
  assumed.

  Args:
    element_name: (`str`) name of the graph element in question.

  Returns:
    (`int`) output slot number.
  """
    _, output_slot = parse_node_or_tensor_name(element_name)
    return output_slot if output_slot is not None else 0