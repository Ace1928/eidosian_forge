import collections
import re
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import remote
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
@staticmethod
def connect(tpu=None, zone=None, project=None):
    """Initializes TPU and returns a TPUClusterResolver.

    This API will connect to remote TPU cluster and initialize the TPU
    hardwares. Example usage:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
    ...     tpu='')

    It can be viewed as convenient wrapper of the following code:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)

    Args:
      tpu: A string corresponding to the TPU to use. It can be the TPU name or
        TPU worker gRPC address. If not set, it will try automatically resolve
        the TPU address on Cloud TPUs.
      zone: Zone where the TPUs are located. If omitted or empty, we will assume
        that the zone of the TPU is the same as the zone of the GCE VM, which we
        will try to discover from the GCE metadata service.
      project: Name of the GCP project containing Cloud TPUs. If omitted or
        empty, we will try to discover the project name of the GCE VM from the
        GCE metadata service.

    Returns:
      An instance of TPUClusterResolver object.

    Raises:
      NotFoundError: If no TPU devices found in eager mode.
    """
    resolver = TPUClusterResolver(tpu, zone, project)
    remote.connect_to_cluster(resolver)
    tpu_strategy_util.initialize_tpu_system_impl(resolver)
    return resolver