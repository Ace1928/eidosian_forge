from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_non_void_return_type(self):

    def f(x, y):
        return x[0] + y[0]
    with self.assertRaisesRegex(TypeError, 'must have void return type'):
        compile_ptx(f, (uint32[::1], uint32[::1]))