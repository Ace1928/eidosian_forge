import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def register_thread(self, thread):
    """Register a thread to join.

    Args:
      thread: A Python thread to join.
    """
    with self._lock:
        self._registered_threads.add(thread)