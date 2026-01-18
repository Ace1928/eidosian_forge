from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def task_address(self, job_name, task_index):
    """Returns the address of the given task in the given job.

    Args:
      job_name: The string name of a job in this cluster.
      task_index: A non-negative integer.

    Returns:
      The address of the given task in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
        job = self._cluster_spec[job_name]
    except KeyError:
        raise ValueError('No such job in cluster: %r' % job_name)
    try:
        return job[task_index]
    except KeyError:
        raise ValueError('No task with index %r in job %r' % (task_index, job_name))