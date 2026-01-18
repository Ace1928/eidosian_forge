from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def check_overload(self, kernel, expected):
    x = np.ones(1, dtype=np.int32)
    cuda.jit(kernel)[1, 1](x)
    self.assertEqual(x[0], expected)