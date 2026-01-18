import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_get_const_mem_unspecialized(self):

    @cuda.jit
    def const_fmt_string(val, to_print):
        if to_print:
            print(val)
    const_fmt_string[1, 1](1, False)
    const_fmt_string[1, 1](1.0, False)
    sig_i64 = void(int64, boolean)
    sig_f64 = void(float64, boolean)
    const_mem_size_i64 = const_fmt_string.get_const_mem_size(sig_i64)
    const_mem_size_f64 = const_fmt_string.get_const_mem_size(sig_f64)
    self.assertIsInstance(const_mem_size_i64, int)
    self.assertIsInstance(const_mem_size_f64, int)
    self.assertGreaterEqual(const_mem_size_i64, 6)
    self.assertGreaterEqual(const_mem_size_f64, 4)
    const_mem_size_all = const_fmt_string.get_const_mem_size()
    self.assertEqual(const_mem_size_all[sig_i64.args], const_mem_size_i64)
    self.assertEqual(const_mem_size_all[sig_f64.args], const_mem_size_f64)