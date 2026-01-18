import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def mark_finished(self):
    """Let the queue know that a closure has been successfully executed."""
    with self._queue_lock:
        if self._inflight_closure_count < 1:
            raise AssertionError('There is no inflight closures to mark_finished.')
        self.inflight_closure_count -= 1
        if self._inflight_closure_count == 0:
            self._no_inflight_closure_condition.notify_all()
        if self._queue.empty() and self._inflight_closure_count == 0:
            self._stop_waiting_condition.notify_all()
        self._watchdog.report_closure_done()