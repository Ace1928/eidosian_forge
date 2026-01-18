import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def run_loop(self):
    """Called at 'timer_interval_secs' boundaries."""
    if self._target:
        self._target(*self._args, **self._kwargs)