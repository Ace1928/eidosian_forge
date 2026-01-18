import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def kernel_wrapper(values):
    n = len(values)
    inputs = [np.empty(n, dtype=numpy_support.as_dtype(tp)) for tp in argtypes]
    output = np.empty(n, dtype=numpy_support.as_dtype(restype))
    for i, vs in enumerate(values):
        for v, inp in zip(vs, inputs):
            inp[i] = v
    args = [output] + inputs
    kernel[int(math.ceil(n / 256)), 256](*args)
    return list(output)