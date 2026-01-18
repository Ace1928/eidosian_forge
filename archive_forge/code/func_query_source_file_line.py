import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
def query_source_file_line(self, file_path, lineno):
    """Query the content of a given line in a source file.

    Args:
      file_path: Path to the source file.
      lineno: Line number as an `int`.

    Returns:
      Content of the line as a string.

    Raises:
      ValueError: If no source file is found at the given file_path.
    """
    if not self._source_files:
        raise ValueError('This debug server has not received any source file contents yet.')
    for source_files in self._source_files:
        for source_file_proto in source_files.source_files:
            if source_file_proto.file_path == file_path:
                return source_file_proto.lines[lineno - 1]
    raise ValueError('Source file at path %s has not been received by the debug server', file_path)