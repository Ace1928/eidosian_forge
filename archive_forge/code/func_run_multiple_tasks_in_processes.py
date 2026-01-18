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
@deprecation.deprecated(None, '`run_multiple_tasks_in_processes` is deprecated; any new test requiring multiple processes should use `multi_process_runner` for better support of log printing, streaming, and more functionality.')
def run_multiple_tasks_in_processes(self, cmd_args, cluster_spec):
    """Run `cmd_args` in a process for each task in `cluster_spec`."""
    processes = {}
    for task_type in cluster_spec.keys():
        processes[task_type] = []
        for task_id in range(len(cluster_spec[task_type])):
            p = self._run_task_in_process(cmd_args, cluster_spec, task_type, task_id)
            processes[task_type].append(p)
    return processes