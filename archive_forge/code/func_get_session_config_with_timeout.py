import collections
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.util.tf_export import tf_export
def get_session_config_with_timeout(timeout_in_secs, cluster_def):
    """Returns a session given a timeout and a cluster configuration."""
    config_proto = config_pb2.ConfigProto(operation_timeout_in_ms=timeout_in_secs, cluster_def=cluster_def)
    return config_proto