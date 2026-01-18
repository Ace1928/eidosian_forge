import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def limit_string_length(string, max_len=50):
    """Limit the length of input string.

  Args:
    string: Input string.
    max_len: (int or None) If int, the length limit. If None, no limit.

  Returns:
    Possibly length-limited string.
  """
    if max_len is None or len(string) <= max_len:
        return string
    else:
        return '...' + string[len(string) - max_len:]