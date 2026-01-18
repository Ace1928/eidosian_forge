import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def float8e4m3_to_float32(ival: int) -> Any:
    if ival < 0 or ival > 255:
        raise ValueError(f'{ival} is not a float8.')
    if ival == 255:
        return np.float32(-np.nan)
    if ival == 127:
        return np.float32(np.nan)
    if ival & 127 == 0:
        return np.float32(0)
    sign = ival & 128
    ival &= 127
    expo = ival >> 3
    mant = ival & 7
    powe = expo & 15
    if expo == 0:
        powe -= 6
        fraction = 0
    else:
        powe -= 7
        fraction = 1
    fval = float(mant / 8 + fraction) * 2.0 ** powe
    if sign:
        fval = -fval
    return np.float32(fval)