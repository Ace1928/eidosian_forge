import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_dim(dim: TensorShapeProto.Dimension) -> str:
    which = dim.WhichOneof('value')
    if which is None:
        return '?'
    return str(getattr(dim, which))