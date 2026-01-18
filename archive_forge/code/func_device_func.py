import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba import cuda, float64
import unittest
def device_func(x, y, z):
    return x * y + z