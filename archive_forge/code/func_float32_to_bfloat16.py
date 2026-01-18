import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def float32_to_bfloat16(fval: float, truncate: bool=False) -> int:
    ival = int.from_bytes(struct.pack('<f', fval), 'little')
    if truncate:
        return ival >> 16
    if isnan(fval):
        return 32704
    rounded = (ival >> 16 & 1) + 32767
    return ival + rounded >> 16