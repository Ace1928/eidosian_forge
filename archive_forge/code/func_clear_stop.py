import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def clear_stop(self):
    """Clears the stop flag.

    After this is called, calls to `should_stop()` will return `False`.
    """
    with self._lock:
        self._joined = False
        self._exc_info_to_raise = None
        if self._stop_event.is_set():
            self._stop_event.clear()