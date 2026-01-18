import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_sub_double_global(self):
    idx = np.random.randint(0, 32, size=32, dtype=np.int64)
    ary = np.zeros(32, np.float64)
    sig = 'void(int64[:], float64[:])'
    cuda_func = cuda.jit(sig)(atomic_sub_double_global)
    cuda_func[1, 32](idx, ary)
    gold = np.zeros(32, dtype=np.float64)
    for i in range(idx.size):
        gold[idx[i]] -= 1.0
    np.testing.assert_equal(ary, gold)