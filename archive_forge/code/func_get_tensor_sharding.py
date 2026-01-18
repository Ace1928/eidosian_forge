import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def get_tensor_sharding(tensor):
    """Returns sharding attribute of a Tensor.

  Args:
    tensor: a Tensor.

  Returns:
    The attribute representing XLA sharding on tensor's op.
  """
    try:
        return get_op_sharding(tensor.op)
    except AttributeError:
        return None