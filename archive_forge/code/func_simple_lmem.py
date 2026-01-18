import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@cuda.jit(void(float32[::1]))
def simple_lmem(ary):
    lm = cuda.local.array(N, dtype=ary.dtype)
    for j in range(N):
        lm[j] = j
    for j in range(N):
        ary[j] = lm[j]