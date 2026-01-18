import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.extract_sub_graph'])
def extract_sub_graph(graph_def, dest_nodes):
    """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

  Args:
    graph_def: A graph_pb2.GraphDef proto.
    dest_nodes: An iterable of strings specifying the destination node names.
  Returns:
    The GraphDef of the sub-graph.

  Raises:
    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
  """
    if not isinstance(graph_def, graph_pb2.GraphDef):
        raise TypeError(f'graph_def must be a graph_pb2.GraphDef proto, but got type {type(graph_def)}.')
    if isinstance(dest_nodes, str):
        raise TypeError(f'dest_nodes must be an iterable of strings, but got type {type(dest_nodes)}.')
    name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(graph_def)
    _assert_nodes_are_present(name_to_node, dest_nodes)
    nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
    out = graph_pb2.GraphDef()
    for n in nodes_to_keep_list:
        out.node.extend([copy.deepcopy(name_to_node[n])])
    out.library.CopyFrom(graph_def.library)
    out.versions.CopyFrom(graph_def.versions)
    return out