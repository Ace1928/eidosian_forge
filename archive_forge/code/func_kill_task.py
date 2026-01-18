import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def kill_task(self, task_type, task_id):
    """Kill a server given task_type and task_id.

    Args:
      task_type: the type of the task such as "worker".
      task_id: the id the task such as 1.
    """
    assert self._mpr
    if not self._start_events[task_type][task_id].is_set() or self._finish_events[task_type][task_id].is_set():
        raise ValueError("The task %s:%d doesn't exist." % (task_type, task_id))
    self._finish_events[task_type][task_id].set()
    self._mpr._processes[task_type, task_id].join()