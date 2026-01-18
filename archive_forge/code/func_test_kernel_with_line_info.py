from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_kernel_with_line_info(self):

    def f():
        pass
    ptx, resty = compile_ptx(f, (), lineinfo=True)
    self.check_line_info(ptx)