from numba import vectorize
from numba import cuda, float32
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def cu_ufunc(x, y, z):
    return cu_device_fn(x, y, z)