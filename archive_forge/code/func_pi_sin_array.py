import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@cuda.jit(void(float32[::1], int64))
def pi_sin_array(x, n):
    i = cuda.grid(1)
    if i < n:
        x[i] = 3.14 * math.sin(x[i])