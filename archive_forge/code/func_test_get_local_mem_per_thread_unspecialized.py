import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_get_local_mem_per_thread_unspecialized(self):
    N = 1000

    @cuda.jit
    def simple_lmem(ary):
        lm = cuda.local.array(N, dtype=ary.dtype)
        for j in range(N):
            lm[j] = j
        for j in range(N):
            ary[j] = lm[j]
    arr_f32 = np.zeros(N, dtype=np.float32)
    arr_f64 = np.zeros(N, dtype=np.float64)
    simple_lmem[1, 1](arr_f32)
    simple_lmem[1, 1](arr_f64)
    sig_f32 = void(float32[::1])
    sig_f64 = void(float64[::1])
    local_mem_f32 = simple_lmem.get_local_mem_per_thread(sig_f32)
    local_mem_f64 = simple_lmem.get_local_mem_per_thread(sig_f64)
    self.assertIsInstance(local_mem_f32, int)
    self.assertIsInstance(local_mem_f64, int)
    self.assertGreaterEqual(local_mem_f32, N * 4)
    self.assertGreaterEqual(local_mem_f64, N * 8)
    local_mem_all = simple_lmem.get_local_mem_per_thread()
    self.assertEqual(local_mem_all[sig_f32.args], local_mem_f32)
    self.assertEqual(local_mem_all[sig_f64.args], local_mem_f64)