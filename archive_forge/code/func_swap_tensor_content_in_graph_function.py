from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
def swap_tensor_content_in_graph_function(graph_def, from_endiness, to_endiness):
    """Fix endiness of tensor contents.

  Args:
    graph_def: Target graph_def to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  """
    if isinstance(graph_def, meta_graph_pb2.MetaGraphDef):
        functions = graph_def.graph_def.library.function
    elif isinstance(graph_def, graph_pb2.GraphDef):
        functions = graph_def.library.function
    else:
        return
    for function in functions:
        node_def = function.node_def
        for node in node_def:
            if node.op == 'Const':
                tensor = node.attr['value'].tensor
                byte_swap_tensor_content(tensor, from_endiness, to_endiness)