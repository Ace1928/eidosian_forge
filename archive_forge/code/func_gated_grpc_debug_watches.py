import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def gated_grpc_debug_watches(self):
    """Get the list of debug watches with attribute gated_grpc=True.

    Since the server receives `GraphDef` from the debugged runtime, it can only
    return such debug watches that it has received so far.

    Returns:
      A `list` of `DebugWatch` `namedtuples` representing the debug watches with
      gated_grpc=True. Each `namedtuple` element has the attributes:
        `node_name` as a `str`,
        `output_slot` as an `int`,
        `debug_op` as a `str`.
    """
    return list(self._gated_grpc_debug_watches)