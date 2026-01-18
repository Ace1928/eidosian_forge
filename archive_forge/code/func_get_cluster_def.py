import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def get_cluster_def(cluster_params, num_workers, num_ps):
    if num_workers > cluster_params.max_num_worker or num_ps > cluster_params.max_num_ps:
        raise ValueError("Requesting more servers than the maximum, adjustcluster params' max_num_ps and max_num_worker")
    if cluster_params.cluster is None:
        cluster_params.cluster = multi_worker_test_base.create_in_process_cluster(num_workers=cluster_params.max_num_worker, num_ps=cluster_params.max_num_ps)
    return {'worker': cluster_params.cluster['worker'][:num_workers], 'ps': cluster_params.cluster['ps'][:num_ps]}