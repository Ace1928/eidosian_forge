import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_ndenumerate_premature_free(self):
    cfunc = njit((types.intp,))(array_ndenumerate_premature_free)
    expect = array_ndenumerate_premature_free(6)
    got = cfunc(6)
    self.assertTrue(got.sum())
    self.assertPreciseEqual(expect, got)