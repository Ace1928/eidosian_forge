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
@contextlib.contextmanager
def skip_if_grpc_server_cant_be_started(test_obj):
    try:
        yield
    except errors.UnknownError as e:
        if 'Could not start gRPC server' in e.message:
            reason = 'Cannot start std servers.'
            test_obj.test_skipped_reason = reason
            test_obj.skipTest(reason)
        else:
            raise