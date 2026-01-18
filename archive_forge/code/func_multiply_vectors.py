import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
from numba.tests.support import skip_unless_cffi
@cuda.jit(link=[functions_cu])
def multiply_vectors(r, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = mul(x[i], y[i])