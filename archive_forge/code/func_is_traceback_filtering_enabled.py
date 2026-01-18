import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.is_traceback_filtering_enabled')
def is_traceback_filtering_enabled():
    """Check whether traceback filtering is currently enabled.

  See also `tf.debugging.enable_traceback_filtering()` and
  `tf.debugging.disable_traceback_filtering()`. Note that filtering out
  internal frames from the tracebacks of exceptions raised by TensorFlow code
  is the default behavior.

  Returns:
    True if traceback filtering is enabled
    (e.g. if `tf.debugging.enable_traceback_filtering()` was called),
    and False otherwise (e.g. if `tf.debugging.disable_traceback_filtering()`
    was called).
  """
    value = getattr(_ENABLE_TRACEBACK_FILTERING, 'value', True)
    return value