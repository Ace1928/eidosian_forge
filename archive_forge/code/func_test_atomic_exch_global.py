import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_exch_global(self):
    rand_const = np.random.randint(50, 100, dtype=np.uint32)
    idx = np.arange(32, dtype=np.uint32)
    ary = np.random.randint(0, 32, size=32, dtype=np.uint32)
    sig = 'void(uint32[:], uint32[:], uint32)'
    cuda_func = cuda.jit(sig)(atomic_exch_global)
    cuda_func[1, 32](idx, ary, rand_const)
    np.testing.assert_equal(ary, rand_const)