import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_get_regs_per_thread_unspecialized(self):

    @cuda.jit
    def pi_sin_array(x, n):
        i = cuda.grid(1)
        if i < n:
            x[i] = 3.14 * math.sin(x[i])
    N = 10
    arr_f32 = np.zeros(N, dtype=np.float32)
    arr_f64 = np.zeros(N, dtype=np.float64)
    pi_sin_array[1, N](arr_f32, N)
    pi_sin_array[1, N](arr_f64, N)
    sig_f32 = void(float32[::1], int64)
    sig_f64 = void(float64[::1], int64)
    regs_per_thread_f32 = pi_sin_array.get_regs_per_thread(sig_f32)
    regs_per_thread_f64 = pi_sin_array.get_regs_per_thread(sig_f64)
    self.assertIsInstance(regs_per_thread_f32, int)
    self.assertIsInstance(regs_per_thread_f64, int)
    self.assertGreater(regs_per_thread_f32, 0)
    self.assertGreater(regs_per_thread_f64, 0)
    regs_per_thread_all = pi_sin_array.get_regs_per_thread()
    self.assertEqual(regs_per_thread_all[sig_f32.args], regs_per_thread_f32)
    self.assertEqual(regs_per_thread_all[sig_f64.args], regs_per_thread_f64)
    if regs_per_thread_f32 == regs_per_thread_f64:
        print('f32 and f64 variant thread usages are equal.')
        print('This may warrant some investigation. Devices:')
        cuda.detect()