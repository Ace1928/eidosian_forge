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
def start_in_process_as(self, as_task_type, as_task_id):
    """Start the processes, with the specified task run in main process.

    This is similar to `start()` except that the task with task_type
    `as_task_type` and task_id `as_task_id` is run in the main process.
    This method is particularly useful when debugging tool such as `pdb` is
    needed in some specific task. Note that since this method is blocking until
    that specific task exits, additional actions would need a thread to be
    called:

    ```python
    def fn():
      # user code to be run
      import pdb; pdb.set_trace()

    def follow_ups():
      time.sleep(5)
      mpr.start_single_process(
          task_type='evaluator',
          task_id=0)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1))
    threading.Thread(target=follow_ups).start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    mpr.join()
    ```

    Note that if `return_output=True`, the logs/stdout by task
    run by the main process is not available in result.stdout.

    Args:
      as_task_type: The task type to be run in the main process.
      as_task_id: The task id to be run in the main process.
    """
    if self._processes:
        raise ValueError('MultiProcessRunner already started.')
    with self._process_lock:
        if self._joined:
            raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
        for task_type, addresses in self._cluster_spec.items():
            for task_id, _ in enumerate(addresses):
                if not (task_type == as_task_type and task_id == as_task_id):
                    self._start_subprocess_and_reading_thread(task_type, task_id)
    _set_tf_config(as_task_type, as_task_id, self._cluster_spec, self._rpc_layer)
    self._fn(*self._args, **self._kwargs)