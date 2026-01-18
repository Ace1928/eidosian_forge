import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@cuda.jit(sig)
def const_array_use(x):
    C = cuda.const.array_like(arr)
    i = cuda.grid(1)
    x[i] = C[i]