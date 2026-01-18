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
def aggregate_and_return_name_for_input(self, out_graphdef):
    """This adds the nodes to out_graphdef and returns an aggregated output.

    In particular, if you have 4 inputs to a hint stub, this will be the
    node that you can use as an output. I.e. you have 4 timesteps from a
    static rnn, then a fused UnidirectionalLSTM will expect 1 input with
    all 4 time steps. So here we make a pack and return the output name of
    that pack.

    Args:
      out_graphdef: A graphdef that is ready to have this input added.

    Returns:
      The name of a pack that aggregates this node.
    """
    flattened = self.flatten_nodes()
    if self.aggregation == OpHint.AGGREGATE_FIRST or self.aggregation == OpHint.AGGREGATE_LAST:
        assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
        return _tensor_name_base(flattened[0].name)
    else:
        new_node = _node_def_pb2.NodeDef()
        new_node.op = 'Pack'
        new_node.name = 'OpHintStack-%s' % flattened[0].name
        new_node.attr['N'].i = len(flattened)
        new_node.attr['T'].type = flattened[0].attr['T'].type
        for discrete in flattened:
            new_node.input.append(_tensor_name_base(discrete.name))
        out_graphdef.node.extend([new_node])
        return new_node.name