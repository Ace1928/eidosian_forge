import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def apply_to_tensor(self, tensor, assign_tuple_sharding=False, use_sharding_op=False, unspecified_dims=None):
    """Applies this Sharding attribute to `tensor`.

    Args:
      tensor: A tf.Tensor to split.
      assign_tuple_sharding: If the sharding type should be a tuple.
      use_sharding_op: Whether to create a sharding op on `tensor`.
      unspecified_dims: An optional list of dimensions unspecified.

    Returns:
      The tensor with Sharding attribute.
    """
    if unspecified_dims:
        assert use_sharding_op and (not assign_tuple_sharding)
    proto = self._proto
    if use_sharding_op:
        if assign_tuple_sharding:
            proto = self._create_tuple_proto(num_outputs=1)
            tensor = tf2xla.sharding(tensor, sharding=proto.SerializeToString())
        else:
            tensor = tf2xla.sharding(tensor, sharding=proto.SerializeToString(), unspecified_dims=unspecified_dims or [])
    elif assign_tuple_sharding or len(tensor.op.outputs) > 1:
        proto = self._get_or_create_tuple_proto(tensor.op)
        tuple_shardings = list(proto.tuple_shardings)
        tuple_shardings[tensor.value_index] = self._proto
        proto = xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=tuple_shardings)
    tensor.op._set_attr('_XlaSharding', attr_value_pb2.AttrValue(s=proto.SerializeToString()))
    return tensor