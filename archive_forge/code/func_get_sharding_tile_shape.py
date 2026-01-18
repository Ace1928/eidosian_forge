import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def get_sharding_tile_shape(sharding):
    """Returns the tile assignment shape for a sharded Tensor.

  Args:
    sharding: a serialized OpSharding message describing the layout of a
      sharded Tensor.

  Returns:
    A list, for each dimension of the sharded Tensor, of the number of shards
      into which it has been split. Returns None if the input indicates no tile
      assignments.
  """
    if sharding is None:
        return None
    sharding_message = xla_data_pb2.OpSharding()
    sharding_message.ParseFromString(sharding)
    if sharding_message.tile_assignment_dimensions:
        return sharding_message.tile_assignment_dimensions
    else:
        return None