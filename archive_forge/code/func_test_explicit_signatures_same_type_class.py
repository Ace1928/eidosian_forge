import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_explicit_signatures_same_type_class(self):
    sigs = ['(float64[::1], float32, float32)', '(float64[::1], float64, float64)']
    f = cuda.jit(sigs)(add_kernel)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, np.float32(1), np.float32(2 ** (-25)))
    self.assertPreciseEqual(r[0], 1.0)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, 1, 2 ** (-25))
    self.assertPreciseEqual(r[0], 1.0000000298023224)