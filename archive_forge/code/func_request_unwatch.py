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
def request_unwatch(self, node_name, output_slot, debug_op):
    """Request disabling a debug tensor watchpoint or breakpoint.

    This is the opposite of `request_watch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
    """
    self._debug_ops_state_change_queue.put(_state_change(debug_service_pb2.EventReply.DebugOpStateChange.DISABLED, node_name, output_slot, debug_op))