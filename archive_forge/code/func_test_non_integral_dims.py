from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_non_integral_dims(self):
    kernfunc = cuda.jit(noop)
    with self.assertRaises(TypeError) as raises:
        kernfunc[2.0, 3]
    self.assertIn('griddim must be a sequence of integers, got [2.0]', str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        kernfunc[2, 3.0]
    self.assertIn('blockdim must be a sequence of integers, got [3.0]', str(raises.exception))