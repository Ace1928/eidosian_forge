from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
@unittest.skip('Requires PR #6367')
def test_issue_6505(self):
    ary = cuda.mapped_array(2, dtype=np.int32)
    ary[:] = 0
    ary_v = ary.view('u1')
    ary_v[1] = 1
    ary_v[5] = 1
    self.assertEqual(sum(ary), 512)