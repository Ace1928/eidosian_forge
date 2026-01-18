import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_dec_global2_64(self):
    rand_const, ary = self.inc_dec_2dim_setup(np.uint64)
    sig = 'void(uint64[:,:], uint64)'
    self.check_dec(ary, rand_const, sig, 1, (4, 8), atomic_dec_global_2)