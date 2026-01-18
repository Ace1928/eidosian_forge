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
@tf_export(v1=['graph_util.must_run_on_cpu'])
def must_run_on_cpu(node, pin_variables_on_cpu=False):
    """Returns True if the given node_def must run on CPU, otherwise False.

  Args:
    node: The node to be assigned to a device. Could be either an ops.Operation
      or NodeDef.
    pin_variables_on_cpu: If True, this function will return False if node_def
      represents a variable-related op.

  Returns:
    True if the given node must run on CPU, otherwise False.
  """
    if isinstance(node, ops.Operation):
        node_def = node.node_def
    else:
        assert isinstance(node, node_def_pb2.NodeDef)
        node_def = node
    if pin_variables_on_cpu and _is_variable_op(node_def.op):
        return True
    if node_def.op == 'Const':
        dtype = node_def.attr['dtype'].type
        if dtype == dtypes.string or dtype == dtypes.int32:
            return True
    if node_def.op in ['DynamicStitch', 'ParallelDynamicStitch']:
        dtype = node_def.attr['T'].type
        if dtype == dtypes.int32:
            return True
    if node_def.op in ['Cast']:
        dtype = node_def.attr['SrcT'].type
        if dtype == dtypes.int32:
            return True
    return False