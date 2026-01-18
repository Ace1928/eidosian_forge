import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
@property
def proto(self):
    """Return the sharding protobuf of type xla_data_pb2.OpSharding."""
    return self._proto