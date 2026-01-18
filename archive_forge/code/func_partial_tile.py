import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
@classmethod
def partial_tile(cls, tile_assignment):
    """Returns a partially tiled sharding attribute.

    This is similar to tile(), but tile_assignment has one more dimension than
    the tensor, and tiles in the last dimension of tile_assignment are
    replicated.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type.
    """
    if not isinstance(tile_assignment, _np.ndarray):
        raise TypeError('PartialTile assignment must be of type np.ndarray')
    dims = list(tile_assignment.shape)
    flattened_devices = tile_assignment.reshape(-1, order='C')
    return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=dims, tile_assignment_devices=list(flattened_devices), replicate_on_last_tile_dim=True))