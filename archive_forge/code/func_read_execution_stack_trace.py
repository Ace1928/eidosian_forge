import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_execution_stack_trace(self, execution):
    """Read the stack trace of a given Execution object.

    Args:
      execution: The Execution object of interest.

    Returns:
      1. The host name.
      2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
    host_name = self._stack_frame_by_id[execution.stack_frame_ids[0]][0]
    return (host_name, [self._stack_frame_by_id[frame_id][1:] for frame_id in execution.stack_frame_ids])