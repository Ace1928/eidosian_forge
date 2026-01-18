import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def get_op_sharding(op):
    """Returns sharding attribute of an op.

  Args:
    op: a TensorFlow op.

  Returns:
    The attribute representing XLA sharding on this op.
  """
    try:
        return op.get_attr('_XlaSharding')
    except ValueError:
        return None
    except AttributeError:
        return None