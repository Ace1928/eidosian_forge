from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def create_local_server(config=None, start=True):
    """Creates a new single-process cluster running on the local host.

    This method is a convenience wrapper for creating a
    `tf.distribute.Server` with a `tf.train.ServerDef` that specifies a
    single-process cluster containing a single task in a job called
    `"local"`.

    Args:
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Returns:
      A local `tf.distribute.Server`.
    """
    return Server({'localhost': ['localhost:0']}, protocol='grpc', config=config, start=start)