import numpy as np
from numba import cuda, float64, void
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import config
@cuda.jit(float64(float64, float64), device=True, inline=True)
def get_max(a, b):
    if a > b:
        return a
    else:
        return b