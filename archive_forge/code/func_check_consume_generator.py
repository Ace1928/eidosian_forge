import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_consume_generator(self, gen_func):
    cgen = jit(nopython=True)(gen_func)
    cfunc = jit(nopython=True)(make_consumer(cgen))
    pyfunc = make_consumer(gen_func)
    expected = pyfunc(5)
    got = cfunc(5)
    self.assertPreciseEqual(got, expected)