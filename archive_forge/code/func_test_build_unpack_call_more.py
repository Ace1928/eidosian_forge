import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_build_unpack_call_more(self):

    def check(p):

        @jit
        def inner(*args):
            return args
        pyfunc = lambda a: inner(1, *a, *(1, 2), *a)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(p), pyfunc(p))
    check((4, 5))
    check((4, 5.5))