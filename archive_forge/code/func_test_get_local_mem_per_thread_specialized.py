import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_get_local_mem_per_thread_specialized(self):
    N = 1000

    @cuda.jit(void(float32[::1]))
    def simple_lmem(ary):
        lm = cuda.local.array(N, dtype=ary.dtype)
        for j in range(N):
            lm[j] = j
        for j in range(N):
            ary[j] = lm[j]
    local_mem_per_thread = simple_lmem.get_local_mem_per_thread()
    self.assertIsInstance(local_mem_per_thread, int)
    self.assertGreaterEqual(local_mem_per_thread, N * 4)