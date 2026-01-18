import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
def scaled_sinc(x, scale):
    if x == 0.0:
        return scale
    else:
        return scale * (math.sin(x * pi) / (pi * x))