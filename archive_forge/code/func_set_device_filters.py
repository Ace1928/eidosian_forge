from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def set_device_filters(self, job_name, task_index, device_filters):
    """Set the device filters for given job name and task id."""
    assert all((isinstance(df, str) for df in device_filters))
    self._device_filters.setdefault(job_name, {})
    self._device_filters[job_name][task_index] = [df for df in device_filters]
    self._cluster_device_filters = None