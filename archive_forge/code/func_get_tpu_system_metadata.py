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
def get_tpu_system_metadata(self):
    """Returns the metadata of the TPU system.

    Users can call this method to get some facts of the TPU system, like
    total number of cores, number of TPU workers and the devices. E.g.
    ```python

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tpu_system_metadata = resolver.get_tpu_system_metadata()
    num_hosts = tpu_system_metadata.num_hosts
    ```

    Returns:
      A `tf.tpu.experimental.TPUSystemMetadata` object.
    """
    cluster_spec = self.cluster_spec()
    cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
    tpu_system_metadata = tpu_system_metadata_lib._query_tpu_system_metadata(self.master(), cluster_def=cluster_def, query_topology=False)
    return tpu_system_metadata