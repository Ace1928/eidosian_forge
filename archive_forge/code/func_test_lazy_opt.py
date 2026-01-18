import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba import cuda, float64
import unittest
def test_lazy_opt(self):
    kernel = cuda.jit(kernel_func)
    x = np.zeros(1, dtype=np.float64)
    kernel[1, 1](x)
    ptx = next(iter(kernel.inspect_asm().items()))[1]
    for fragment in removed_by_opt:
        with self.subTest(fragment=fragment):
            self.assertNotIn(fragment, ptx)