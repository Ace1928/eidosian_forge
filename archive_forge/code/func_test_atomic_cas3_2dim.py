import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_cas3_2dim(self):
    rfill = np.random.randint(50, 500, dtype=np.uint32)
    runfill = np.random.randint(1, 25, dtype=np.uint32)
    self.check_cas(n=100, fill=rfill, unfill=runfill, dtype=np.uint32, cas_func=atomic_cas_2dim, ndim=2)