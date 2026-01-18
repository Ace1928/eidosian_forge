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
def terminate(self, task_type, task_id):
    """Terminates the process with `task_type` and `task_id`.

    If auto_retart=True, the terminated task will be restarted unless the chief
    has already exited with zero exit code.

    Args:
      task_type: the task type.
      task_id: the task id.

    """
    with self._process_lock:
        p = self._processes.get((task_type, task_id), None)
        if p is None:
            raise ValueError('{}-{} does not exist'.format(task_type, task_id))
        self._terminated.add((task_type, task_id))
        self._parent_to_sub_queue.put('terminate {} {}'.format(task_type, task_id))
        p.join()