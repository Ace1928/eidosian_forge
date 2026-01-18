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
@tf_export('__internal__.distribute.multi_process_runner.create_cluster_spec', v1=[])
def create_cluster_spec(has_chief=False, num_workers=1, num_ps=0, has_eval=False):
    """Create a cluster spec with tasks with unused local ports.

  This utility finds available ports at localhost, and returns a dict that
  represents the cluster spec that utilizes those ports, according to the
  arguments. The dict representing the cluster spec contains task types, and
  their instances' addresses. Note that this is usually only for testing purpose
  using multiple processes in the local machine, and should not be used for real
  multi-worker TensorFlow programs, where the addresses need to point to the
  processes at separate machines.

  This util is useful when creating the `cluster_spec` arg for
  `tf.__internal__.distribute.multi_process_runner.run`.

  Args:
    has_chief: Whether the generated cluster spec should contain "chief" task
      type.
    num_workers: Number of workers to use in the cluster spec.
    num_ps: Number of parameter servers to use in the cluster spec.
    has_eval: Whether this cluster spec has evaluator.

  Returns:
    A dict that represents the cluster spec using localhost ports for the tasks.

  Example:

  ```python
  cluster_spec =
  tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
      has_chief=True, num_workers=2, num_ps=2)
  # An example of cluster_spec is
  # {'chief': ['localhost:23381'],
  # 'worker': ['localhost:19197', 'localhost:22903'],
  # 'ps': ['localhost:16912', 'localhost:21535']}

  cluster_spec =
  tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
      has_chief=False, num_workers=0, num_ps=0, has_eval=True)
  # An example of cluster_spec is
  # {'evaluator': ['localhost:23381']}
  ```
  """
    cluster_spec = {}
    if has_chief:
        cluster_spec['chief'] = ['localhost:%s' % pick_unused_port()]
    if num_workers:
        cluster_spec['worker'] = ['localhost:%s' % pick_unused_port() for _ in range(num_workers)]
    if num_ps:
        cluster_spec['ps'] = ['localhost:%s' % pick_unused_port() for _ in range(num_ps)]
    if has_eval:
        cluster_spec['evaluator'] = ['localhost:%s' % pick_unused_port()]
    return cluster_spec