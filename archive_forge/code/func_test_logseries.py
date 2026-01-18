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
def test_logseries(self):
    test_sizes = [None, (), (100,), (10, 20, 30)]
    bitgen_types = [None, MT19937]
    dist_func = lambda x, size, dtype: x.logseries(0.3, size=size)
    for _size, _bitgen in itertools.product(test_sizes, bitgen_types):
        with self.subTest(_size=_size, _bitgen=_bitgen):
            self.check_numpy_parity(dist_func, _bitgen, None, _size, None)
    dist_func = lambda x, p, size: x.logseries(p=p, size=size)
    valid_args = [0.3, (1,)]
    self._check_invalid_types(dist_func, ['p', 'size'], valid_args, ['x', ('x',)])
    rng = np.random.default_rng(1)
    valid_args = [rng] + valid_args
    nb_dist_func = numba.njit(dist_func)
    for _p in [-0.1, 1, np.nan]:
        with self.assertRaises(ValueError) as raises:
            curr_args = valid_args.copy()
            curr_args[1] = _p
            nb_dist_func(*curr_args)
        self.assertIn('p < 0, p >= 1 or p is NaN', str(raises.exception))
    self.disable_leak_check()