import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_no_double_specialize_no_sig_diff_types(self):

    @cuda.jit
    def f(x):
        pass
    f_specialized = f.specialize(int32[::1])
    self._test_no_double_specialize(f_specialized, float32[::1])