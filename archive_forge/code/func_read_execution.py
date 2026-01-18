import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_execution(self, execution_digest):
    """Read a detailed Execution object."""
    debug_event = self._reader.read_execution_event(execution_digest.locator)
    return _execution_from_debug_event_proto(debug_event, execution_digest.locator)