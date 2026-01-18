import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@property
def missing_devices(self):
    """Array of indices of missing devices."""
    return self._missing_devices