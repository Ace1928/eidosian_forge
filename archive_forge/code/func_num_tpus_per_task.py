import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@property
def num_tpus_per_task(self):
    """Returns the number of TPU devices per task in the TPU slice."""
    return self._device_coordinates.shape[1]