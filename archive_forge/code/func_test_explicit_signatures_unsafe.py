import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('Simulator does not use _prepare_args')
@unittest.expectedFailure
def test_explicit_signatures_unsafe(self):
    f = cuda.jit('(int64[::1], int64, int64)')(add_kernel)
    r = np.zeros(1, dtype=np.int64)
    f[1, 1](r, 1.5, 2.5)
    self.assertPreciseEqual(r[0], 3)
    self.assertEqual(len(f.overloads), 1, f.overloads)
    sigs = ['(int64[::1], int64, int64)', '(float64[::1], float64, float64)']
    f = cuda.jit(sigs)(add_kernel)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, np.int32(1), 2.5)
    self.assertPreciseEqual(r[0], 3.5)