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
def task_function(start_events, finish_events):
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    cluster_spec = cluster_resolver.cluster_spec()
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    rpc_layer = cluster_resolver.rpc_layer
    server_config = config_pb2.ConfigProto()
    server_config.device_count['GPU'] = 0
    if collective_leader:
        server_config.experimental.collective_group_leader = collective_leader
        server_config.experimental.collective_nccl = False
        logging.info('Enabling collective ops with cluster_spec = %r, task_type = %r, task_id = %r, rpc_layer = %r, collective_leader = %s', cluster_spec, task_type, task_id, rpc_layer, collective_leader)
    else:
        logging.info('Starting server with cluster_spec = %r, task_type = %r, task_id = %r, rpc_layer = %r', cluster_spec, task_type, task_id, rpc_layer)
    server_lib.Server(cluster_spec, job_name=task_type, protocol=rpc_layer, task_index=task_id, config=server_config, start=True)
    start_event = start_events[task_type][task_id]
    start_event.set()
    finish_event = finish_events[task_type][task_id]
    finish_event.wait()
    os._exit(0)