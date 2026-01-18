import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def auto_to_manual_spmd_partition(tensor, manual_sharding, single_dim=-1, unspecified_dims=None):
    """Switches from automatic SPMD partitioning to manual partitioning.

  Converts a full-shaped tensor (to be automatically partitioned by SPMD
  partitioner) to a shard-shaped tensor to be consumed by manually partitioned
  ops.

  Args:
    tensor: A tf.Tensor in full shape.
    manual_sharding: A serialized string of OpSharding to be used in manual
      partitioning.
    single_dim: If >= 0, the conversion will happen only on this dim in
      subgroups.
    unspecified_dims: An optional list of dimensions unspecified.

  Returns:
    A shard-shaped tensor to be consumed by manually partitioned ops.
  """
    return tf2xla.spmd_full_to_shard_shape(tensor, manual_sharding=manual_sharding, dim=single_dim, unspecified_dims=unspecified_dims or [])