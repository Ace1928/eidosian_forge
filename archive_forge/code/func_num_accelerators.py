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
def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    """Returns the number of TPU cores per worker.

    Connects to the master and list all the devices present in the master,
    and counts them up. Also verifies that the device counts per host in the
    cluster is the same before returning the number of TPU cores per host.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Used to create a connection to a TPU master in order to
        retrieve the system metadata.

    Raises:
      RuntimeError: If we cannot talk to a TPU worker after retrying or if the
        number of TPU devices per host is different.
    """
    if self._tpu == 'local':
        return {'TPU': len([d for d in framework_config.list_logical_devices() if d.device_type == 'TPU'])}
    retry_count = 1
    while True:
        try:
            device_details = TPUClusterResolver._get_device_dict_and_cores(cluster_resolver_lib.get_accelerator_devices(self.master(), config_proto=config_proto))
            break
        except errors.DeadlineExceededError:
            error_message = 'Failed to connect to master. The TPU might not be ready (e.g. still scheduling) or the master address is incorrect: got (%s)' % self.master()
            if retry_count <= _TPU_CONN_RETRIES:
                logging.warning(error_message)
                logging.warning('Retrying (%d/%d)...', retry_count, _TPU_CONN_RETRIES)
                retry_count += 1
            else:
                raise RuntimeError(error_message)
    if device_details.total_cores:
        return {'TPU': TPUClusterResolver._verify_and_return_same_core_count(device_details.device_map)}
    return {'TPU': 0}