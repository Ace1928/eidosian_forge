import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def get_trtengineop_node_op_count(graphdef, node_name):
    """Counts the number of nodes and OP types of a given TRTEngineOp."""
    ops_in_engine = collections.defaultdict(int)
    for func in graphdef.library.function:
        if f'{node_name}_native_segment' == func.signature.name:
            node_count = len(func.node_def)
            for node in func.node_def:
                ops_in_engine[node.op] += 1
            break
    return (node_count, ops_in_engine)