import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def execution_to_tensor_values(self, execution):
    """Read the full tensor values from an Execution or ExecutionDigest.

    Args:
      execution: An `ExecutionDigest` or `ExeuctionDigest` object.

    Returns:
      A list of numpy arrays representing the output tensor values of the
        execution event.
    """
    debug_event = self._reader.read_execution_event(execution.locator)
    return [_parse_tensor_value(tensor_proto) for tensor_proto in debug_event.execution.tensor_protos]