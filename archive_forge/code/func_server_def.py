from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@property
def server_def(self):
    """Returns the `tf.train.ServerDef` for this server.

    Returns:
      A `tf.train.ServerDef` protocol buffer that describes the configuration
      of this server.
    """
    return self._server_def