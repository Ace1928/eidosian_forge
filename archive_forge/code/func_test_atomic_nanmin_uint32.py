import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_nanmin_uint32(self):
    self.check_atomic_nanmin(dtype=np.uint32, lo=0, hi=65535, init_val=0)