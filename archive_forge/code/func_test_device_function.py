from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_device_function(self):

    def add(x, y):
        return x + y
    args = (float32, float32)
    ptx, resty = compile_ptx(add, args, device=True)
    self.assertIn('func_retval', ptx)
    self.assertIn('.visible .func', ptx)
    self.assertNotIn('.visible .entry', ptx)
    self.assertEqual(resty, float32)
    sig_int32 = int32(int32, int32)
    ptx, resty = compile_ptx(add, sig_int32, device=True)
    self.assertEqual(resty, int32)
    sig_int16 = int16(int16, int16)
    ptx, resty = compile_ptx(add, sig_int16, device=True)
    self.assertEqual(resty, int16)
    sig_string = 'uint32(uint32, uint32)'
    ptx, resty = compile_ptx(add, sig_string, device=True)
    self.assertEqual(resty, uint32)