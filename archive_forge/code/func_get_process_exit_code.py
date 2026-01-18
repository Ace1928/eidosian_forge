import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
def get_process_exit_code(self, task_type, task_id):
    """Returns the subprocess exit code given the task type and task id.

    Args:
      task_type: The task type.
      task_id: The task id.

    Returns:
      The subprocess exit code; `None` if the subprocess has not exited yet.

    Raises:
      KeyError: If the corresponding subprocess is not found with `task_type`
        and `task_id`.
    """
    with self._process_lock:
        p = self._processes[task_type, task_id]
    return p.exitcode if p else None