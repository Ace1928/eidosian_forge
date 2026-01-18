import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def bfloat16_to_float32(ival: int) -> Any:
    if ival == 32704:
        return np.float32(np.nan)
    expo = ival >> 7
    prec = ival - (expo << 7)
    sign = expo & 256
    powe = expo & 255
    fval = float(prec * 2 ** (-7) + 1) * 2.0 ** (powe - 127)
    if sign:
        fval = -fval
    return np.float32(fval)