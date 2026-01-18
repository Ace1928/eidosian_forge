import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_cas2_2dim(self):
    self.check_cas(n=100, fill=-45, unfill=-1, dtype=np.int64, cas_func=atomic_cas_2dim, ndim=2)