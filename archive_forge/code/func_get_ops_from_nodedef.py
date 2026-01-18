import json
import os
import sys
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_kernel_registry
def get_ops_from_nodedef(node_def):
    """Gets the op and kernel needed from the given NodeDef.

  Args:
    node_def: TF NodeDef to get op/kernel information.

  Returns:
    A tuple of (op_name, kernel_name). If the op is not in the allowlist of ops
    without kernel and there is no kernel found, then return None.
  """
    if not node_def.device:
        node_def.device = '/cpu:0'
    kernel_class = _pywrap_kernel_registry.TryFindKernelClass(node_def.SerializeToString())
    op = str(node_def.op)
    if kernel_class or op in OPS_WITHOUT_KERNEL_ALLOWLIST:
        return (op, str(kernel_class.decode('utf-8')) if kernel_class else None)
    else:
        tf_logging.warning('Warning: no kernel found for op %s', op)
        return None