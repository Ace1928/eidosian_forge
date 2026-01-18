import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_source_files_event(self, offset):
    """Read a DebugEvent proto at given offset from the .source_files file."""
    with self._reader_read_locks[self._source_files_path]:
        proto_string = self._get_reader(self._source_files_path).read(offset)[0]
    return debug_event_pb2.DebugEvent.FromString(proto_string)