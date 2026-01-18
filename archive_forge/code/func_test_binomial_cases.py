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
def test_binomial_cases(self):
    cases = [(1, 0.1), (50, 0.9), (100, 0.4), (100, 0.9)]
    size = None
    for n, p in cases:
        with self.subTest(n=n, p=p):
            dist_func = lambda x, size, dtype: x.binomial(n, p, size=size)
            self.check_numpy_parity(dist_func, None, None, size, None, 0)