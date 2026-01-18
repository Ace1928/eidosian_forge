import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@property
def mesh_shape(self):
    """A rank 1 int32 array describing the shape of the TPU topology."""
    return self._mesh_shape