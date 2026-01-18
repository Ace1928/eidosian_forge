import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def source_lines(self, host_name, file_path):
    """Read the line-by-line content of a source file.

    Args:
      host_name: Host name on which the source file is located.
      file_path: File path at which the source file is located.

    Returns:
      Lines of the source file as a `list` of `str`s.
    """
    offset = self._host_name_file_path_to_offset[host_name, file_path]
    return list(self._reader.read_source_files_event(offset).source_file.lines)