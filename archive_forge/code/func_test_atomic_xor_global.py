import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_xor_global(self):
    rand_const = np.random.randint(500)
    idx = np.random.randint(0, 32, size=32, dtype=np.int32)
    ary = np.random.randint(0, 32, size=32, dtype=np.int32)
    gold = ary.copy()
    sig = 'void(int32[:], int32[:], int32)'
    cuda_func = cuda.jit(sig)(atomic_xor_global)
    cuda_func[1, 32](idx, ary, rand_const)
    for i in range(idx.size):
        gold[idx[i]] ^= rand_const
    np.testing.assert_equal(ary, gold)