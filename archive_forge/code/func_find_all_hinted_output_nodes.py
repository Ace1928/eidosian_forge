import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
def find_all_hinted_output_nodes(session=None, graph_def=None):
    """Find all Ophints output nodes in the graph.

  This is used to get all the output nodes those are ophinted, it is important
  for operation like convert_variables_to_constants keep all ophints structure.
  Note: only one of session or graph_def should be used, not both.
  Why this can be useful? Some TensorFlow ops (e.g. bidirectional rnn), can
  generate multiple outputs for unfused subgraph. If not all output nodes are
  consumed, graph optimization can potentially drop the unused nodes and cause
  ophints in an invalid states (due to missing ophinted output nodes). So it's
  important for us to find all those hinted output nodes and make sure they're
  not discarded away.

  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.

  Returns:
    A list of OpHints output nodes.
  Raises:
    ValueError: If both session and graph_def are provided.
  """
    if session is not None and graph_def is not None:
        raise ValueError('Provide only one of session and graph_def.')
    hinted_outputs_nodes = []
    if session is not None:
        hints = _find_all_hints_in_nodes(session.graph_def.node)
    elif graph_def is not None:
        hints = _find_all_hints_in_nodes(graph_def.node)
    for hint in hints.values():
        _, output_nodes = hint.flattened_inputs_and_outputs()
        hinted_outputs_nodes.extend(output_nodes)
    return hinted_outputs_nodes