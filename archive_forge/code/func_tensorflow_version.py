import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def tensorflow_version(self):
    """TensorFlow version used in the debugged TensorFlow program.

    Note: this is not necessarily the same as the version of TensorFlow used to
    load the DebugEvent file set.

    Returns:
      TensorFlow version used by the debugged program, as a `str`.
    """
    return self._reader.tensorflow_version()