import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_gufunc_scalar_output_bug(self):

    @guvectorize(['void(int32, int32[:])'], '()->()', target='cuda')
    def twice(inp, out):
        out[0] = inp * 2
    self.assertEqual(twice(10), 20)
    arg = np.arange(10).astype(np.int32)
    self.assertPreciseEqual(twice(arg), arg * 2)