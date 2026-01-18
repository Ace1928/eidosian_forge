from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
@cuda.jit(device=True)
def dev_func(x):
    return floor(x)