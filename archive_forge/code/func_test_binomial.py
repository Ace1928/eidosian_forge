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
def test_binomial(self):
    test_sizes = [None, (), (100,), (10, 20, 30)]
    bitgen_types = [None, MT19937]
    dist_func = lambda x, size, dtype: x.binomial(n=1, p=0.1, size=size)
    for _size in test_sizes:
        for _bitgen in bitgen_types:
            with self.subTest(_size=_size, _bitgen=_bitgen):
                self.check_numpy_parity(dist_func, _bitgen, None, _size, None, adjusted_ulp_prec)
    dist_func = lambda x, n, p, size: x.binomial(n=n, p=p, size=size)
    self._check_invalid_types(dist_func, ['n', 'p', 'size'], [1, 0.75, (1,)], ['x', 'x', ('x',)])