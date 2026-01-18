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
def create_in_process_cluster(num_workers, num_ps, has_chief=False, has_eval=False, rpc_layer='grpc'):
    """Create an in-process cluster that consists of only standard server."""
    gpu_mem_frac = 0.7 / (num_workers + int(has_chief) + int(has_eval))
    worker_config = config_pb2.ConfigProto()
    worker_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
    if worker_config.inter_op_parallelism_threads < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1
    if has_chief:
        worker_config.experimental.collective_group_leader = '/job:chief/replica:0/task:0'
    else:
        worker_config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    ps_config = config_pb2.ConfigProto()
    ps_config.device_count['GPU'] = 0
    eval_config = config_pb2.ConfigProto()
    eval_config.experimental.collective_group_leader = ''
    cluster = None
    try:
        cluster = _create_cluster(num_workers, num_ps=num_ps, has_chief=has_chief, has_eval=has_eval, worker_config=worker_config, ps_config=ps_config, eval_config=eval_config, protocol=rpc_layer)
    except errors.UnknownError as e:
        if 'Could not start gRPC server' in e.message:
            raise unittest.SkipTest('Cannot start std servers.')
        else:
            raise
    return cluster