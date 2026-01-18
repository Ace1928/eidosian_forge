import numba
import numpy as np
import sys
import itertools
import gc
from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError
from numba.tests.support import run_in_new_process_caching, SerialMixin
def test_integers_cases(self):
    cases = [(5, 6, np.uint64), (5, 100, np.uint64), (0, 1099511627775, np.uint64), (0, 18446744073709551615 - 1, np.uint64), (0, 18446744073709551615, np.uint64), (5, 6, np.int64), (5, 100, np.int64), (0, 1099511627775, np.int64), (0, 1152921504606846975 - 1, np.int64), (0, 1152921504606846975, np.int64), (-1152921504606846975, 1152921504606846975, np.int64), (5, 6, np.uint32), (5, 100, np.uint32), (0, 4294967295 - 1, np.uint32), (0, 4294967295, np.uint32), (5, 6, np.int32), (5, 100, np.int32), (0, 268435455 - 1, np.int32), (0, 268435455, np.int32), (-268435455, 268435455, np.int32), (5, 6, np.uint16), (5, 100, np.uint16), (0, 65535 - 1, np.uint16), (0, 65535, np.uint16), (5, 6, np.int16), (5, 10, np.int16), (0, 4095 - 1, np.int16), (0, 4095, np.int16), (-4095, 4095, np.int16), (5, 6, np.uint8), (5, 10, np.uint8), (0, 255 - 1, np.uint8), (0, 255, np.uint8), (5, 6, np.int8), (5, 10, np.int8), (0, 15 - 1, np.int8), (0, 15, np.int8), (-15, 15, np.int8)]
    size = (2, 3)
    for low, high, dtype in cases:
        with self.subTest(low=low, high=high, dtype=dtype):
            dist_func = lambda x, size, dtype: x.integers(low, high, size=size, dtype=dtype)
            self.check_numpy_parity(dist_func, None, None, size, dtype, 0)