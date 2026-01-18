from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, skip_unless_cc_53
def test_is_fp16_supported(self):
    self.assertTrue(cuda.is_float16_supported())