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
def format_args(items):
    s = ''
    for idx, item in items.iteritems():
        s += '\t\t%d:\n' % idx + str(item)
    return s