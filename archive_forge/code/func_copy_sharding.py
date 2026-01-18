import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def copy_sharding(from_tensor, to_tensor, use_sharding_op=False):
    """Copies the a tensor's sharding to another.

  Args:
    from_tensor: Source tensor. Must be the sole output of an op.
    to_tensor: the tensor the annotate with the copy.
    use_sharding_op: whether to create a sharding op on `to_tensor`.

  Returns:
    A tensor with sharding annotation copied from `from_tensor`.
  """
    sharding = get_tensor_sharding(from_tensor)
    if sharding is None:
        return to_tensor
    if use_sharding_op:
        to_tensor = tf2xla.sharding(to_tensor, sharding=sharding)
    attr_value = attr_value_pb2.AttrValue(s=sharding)
    to_tensor.op._set_attr('_XlaSharding', attr_value)
    return to_tensor