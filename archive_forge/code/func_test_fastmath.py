from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_fastmath(self):

    def f(x, y, z, d):
        return sqrt((x * y + z) / d)
    args = (float32, float32, float32, float32)
    ptx, resty = compile_ptx(f, args, device=True)
    self.assertIn('fma.rn.f32', ptx)
    self.assertIn('div.rn.f32', ptx)
    self.assertIn('sqrt.rn.f32', ptx)
    ptx, resty = compile_ptx(f, args, device=True, fastmath=True)
    self.assertIn('fma.rn.ftz.f32', ptx)
    self.assertIn('div.approx.ftz.f32', ptx)
    self.assertIn('sqrt.approx.ftz.f32', ptx)