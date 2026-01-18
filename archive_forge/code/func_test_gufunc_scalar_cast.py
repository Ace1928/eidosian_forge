import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_gufunc_scalar_cast(self):

    @guvectorize(['void(int32, int32[:], int32[:])'], '(),(t)->(t)', target='cuda')
    def foo(a, b, out):
        for i in range(b.size):
            out[i] = a * b[i]
    a = np.int64(2)
    b = np.arange(10).astype(np.int32)
    out = foo(a, b)
    np.testing.assert_equal(out, a * b)
    a = np.array(a)
    da = cuda.to_device(a)
    self.assertEqual(da.dtype, np.int64)
    with self.assertRaises(TypeError) as raises:
        foo(da, b)
    self.assertIn('does not support .astype()', str(raises.exception))