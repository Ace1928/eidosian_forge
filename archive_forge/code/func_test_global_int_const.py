import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import unittest, CUDATestCase
def test_global_int_const(self):
    """Test simple_smem
        """
    compiled = cuda.jit('void(int32[:])')(simple_smem)
    nelem = 100
    ary = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary)
    self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))