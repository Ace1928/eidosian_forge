import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_explicit_signatures_mixed(self):
    sigs = [(int64[::1], int64, int64), '(float64[::1], float64, float64)']
    self._test_explicit_signatures(sigs)
    sigs = [(int64[::1], int64, int64), void(float64[::1], float64, float64)]
    self._test_explicit_signatures(sigs)
    sigs = [void(int64[::1], int64, int64), '(float64[::1], float64, float64)']
    self._test_explicit_signatures(sigs)