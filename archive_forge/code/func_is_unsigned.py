import numpy as np
from . import pywrap_tensorflow
from tensorboard.compat.proto import types_pb2
@property
def is_unsigned(self):
    """Returns whether this type is unsigned.

        Non-numeric, unordered, and quantized types are not considered unsigned, and
        this function returns `False`.

        Returns:
          Whether a `DType` is unsigned.
        """
    try:
        return self.min == 0
    except TypeError:
        return False