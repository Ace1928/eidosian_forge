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
def request_watch(self, node_name, output_slot, debug_op, breakpoint=False):
    """Request enabling a debug tensor watchpoint or breakpoint.

    This will let the server send a EventReply to the client side
    (i.e., the debugged TensorFlow runtime process) to request adding a watch
    key (i.e., <node_name>:<output_slot>:<debug_op>) to the list of enabled
    watch keys. The list applies only to debug ops with the attribute
    gated_grpc=True.

    To disable the watch, use `request_unwatch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
      breakpoint: (`bool`) Iff `True`, the debug op will block and wait until it
        receives an `EventReply` response from the server. The `EventReply`
        proto may carry a TensorProto that modifies the value of the debug op's
        output tensor.
    """
    self._debug_ops_state_change_queue.put(_state_change(debug_service_pb2.EventReply.DebugOpStateChange.READ_WRITE if breakpoint else debug_service_pb2.EventReply.DebugOpStateChange.READ_ONLY, node_name, output_slot, debug_op))