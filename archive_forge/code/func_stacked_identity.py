import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def stacked_identity(batch_shape, n, dtype):
    shape = batch_shape + (n, n)
    idx = cupy.arange(n)
    x = cupy.zeros(shape, dtype)
    x[..., idx, idx] = 1
    return x