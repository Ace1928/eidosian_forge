import numpy as np
from . import pywrap_tensorflow
from tensorboard.compat.proto import types_pb2
@property
def as_datatype_enum(self):
    """Returns a `types_pb2.DataType` enum value based on this `DType`."""
    return self._type_enum