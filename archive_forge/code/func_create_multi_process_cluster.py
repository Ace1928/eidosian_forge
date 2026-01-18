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
def create_multi_process_cluster(num_workers, num_ps, has_chief=False, has_eval=False, rpc_layer='grpc', stream_output=False, collective_leader=None):
    logging.info(f'Now creating a MultiProcessCluster with num_workers={num_workers}, num_ps={num_ps}.')
    cluster_spec = create_cluster_spec(has_chief=has_chief, num_workers=num_workers, num_ps=num_ps, has_eval=has_eval)
    cluster = MultiProcessCluster(cluster_resolver_lib.SimpleClusterResolver(server_lib.ClusterSpec(cluster_spec), rpc_layer=rpc_layer), stream_output=stream_output, collective_leader=collective_leader)
    cluster.start()
    return cluster