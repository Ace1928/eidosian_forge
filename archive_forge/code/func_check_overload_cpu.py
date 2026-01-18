from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def check_overload_cpu(self, kernel, expected):
    x = np.ones(1, dtype=np.int32)
    njit(kernel)(x)
    self.assertEqual(x[0], expected)