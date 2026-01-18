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
def test_bitgen_funcs(self):
    func_names = ['next_uint32', 'next_uint64', 'next_double']
    funcs = [next_uint32, next_uint64, next_double]
    for _func, _func_name in zip(funcs, func_names):
        with self.subTest(_func=_func, _func_name=_func_name):
            self._test_bitgen_func_parity(_func_name, _func)