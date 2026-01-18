from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def task_indices(self, job_name):
    """Returns a list of valid task indices in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of valid task indices in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
        job = self._cluster_spec[job_name]
    except KeyError:
        raise ValueError('No such job in cluster: %r' % job_name)
    return list(sorted(job.keys()))