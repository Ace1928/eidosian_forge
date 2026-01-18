import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_dyn_shared_memory(self):
    compiled = cuda.jit('void(float32[::1])')(dyn_shared_memory)
    shape = 50
    ary = np.empty(shape, dtype=np.float32)
    compiled[1, shape, 0, ary.size * 4](ary)
    self.assertTrue(np.all(ary == 2 * np.arange(ary.size, dtype=np.int32)))