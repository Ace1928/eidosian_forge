import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def apply_to_operation(self, operation):
    """Applies this Sharding attribute to `operation`.

    Args:
      operation: A tf.Operation to add sharding annotation.
    """
    attr_value = attr_value_pb2.AttrValue(s=self._proto.SerializeToString())
    operation._set_attr('_XlaSharding', attr_value)