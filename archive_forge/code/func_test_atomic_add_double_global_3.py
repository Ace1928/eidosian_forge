import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add_double_global_3(self):
    ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
    orig = ary.copy()
    cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_global_3)
    cuda_func[1, (4, 8)](ary)
    np.testing.assert_equal(ary, orig + 1)
    self.assertCorrectFloat64Atomics(cuda_func, shared=False)