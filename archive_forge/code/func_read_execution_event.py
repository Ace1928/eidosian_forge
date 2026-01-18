import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_execution_event(self, offset):
    """Read a DebugEvent proto at a given offset from the .execution file.

    Args:
      offset: Offset to read the DebugEvent proto from.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
    with self._reader_read_locks[self._execution_path]:
        proto_string = self._get_reader(self._execution_path).read(offset)[0]
    return debug_event_pb2.DebugEvent.FromString(proto_string)