import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_coerce_input_types(self):
    c_add = cuda.jit(add_kernel)
    r = np.zeros(1, dtype=np.complex128)
    c_add[1, 1](r, 123, 456)
    self.assertEqual(r[0], add(123, 456))
    c_add[1, 1](r, 12.3, 45.6)
    self.assertEqual(r[0], add(12.3, 45.6))
    c_add[1, 1](r, 12.3, 45.6j)
    self.assertEqual(r[0], add(12.3, 45.6j))
    c_add[1, 1](r, 12300000000, 456)
    self.assertEqual(r[0], add(12300000000, 456))
    c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
    r = np.zeros(1, dtype=np.int32)
    c_add[1, 1](r, 123, 456)
    self.assertPreciseEqual(r[0], add(123, 456))