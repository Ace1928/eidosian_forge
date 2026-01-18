import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_graph_op_creation_stack_trace(self, graph_op_creation_digest):
    """Read the stack trace of a given graph op creation object.

    Args:
      graph_op_creation_digest: The GraphOpCreationDigest object of interest.

    Returns:
      A tuple consisting of:
        1. The host name.
        2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
    return (graph_op_creation_digest.host_name, [self._stack_frame_by_id[frame_id][1:] for frame_id in graph_op_creation_digest.stack_frame_ids])