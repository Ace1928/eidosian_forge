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
def aggregate_and_return_name_for_output(self, fused_op_name, output_index, out_graphdef):
    """This adds to `out_graphdef` all the unaggregated outputs.

    I.e. we are outputting from a fused stub, but we need to make it compatible
    with the unfused original graph so we insert an unpack. Ideally in a later
    stage the unpack -> pack sequences will be removed.

    Args:
      fused_op_name: The name of the stub we are in the process of fusing.
      output_index: The output output_index this object represents.
      out_graphdef: The graphdef we are in the process of buildings

    Returns:
      The type of the aggregated output (so we can finish building the stub
      op).
    """
    flattened = self.flatten_nodes()
    if self.aggregation == OpHint.AGGREGATE_FIRST or self.aggregation == OpHint.AGGREGATE_LAST:
        assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
        temp_op = _LiteSingleOperand(flattened[0])
        return temp_op.aggregate_and_return_name_for_output(fused_op_name, output_index, out_graphdef)
    else:
        stack_node = _node_def_pb2.NodeDef()
        stack_node.op = 'Unpack'
        stack_node.name = 'OpHintUnstack-%s' % flattened[0].name
        stack_node.attr['num'].i = len(flattened)
        output_type = flattened[0].attr['T'].type
        stack_node.attr['T'].type = output_type
        stack_node.input.append(_tensorflow_output_name(fused_op_name, output_index))
        out_graphdef.node.extend([stack_node])
        for idx, discrete in enumerate(flattened):
            output_node = _copy.deepcopy(discrete)
            del output_node.input[:]
            output_node.input.append(_tensorflow_output_name(stack_node.name, idx))
            out_graphdef.node.extend([output_node])
        return output_type