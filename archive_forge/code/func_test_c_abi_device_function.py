from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_c_abi_device_function(self):

    def f(x, y):
        return x + y
    ptx, resty = compile_ptx(f, int32(int32, int32), device=True, abi='c')
    self.assertNotIn(ptx, 'param_2')
    self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b32\\s+func_retval0\\)\\s+f\\(')
    ptx, resty = compile_ptx(f, int64(int64, int64), device=True, abi='c')
    self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b64')