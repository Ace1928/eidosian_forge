import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_hetero_tuple(self):
    tuple_type = types.Tuple((types.int64, types.float32))
    cf_first = njit((tuple_type,))(tuple_first)
    cf_second = njit((tuple_type,))(tuple_second)
    self.assertPreciseEqual(cf_first((2 ** 61, 1.5)), 2 ** 61)
    self.assertPreciseEqual(cf_second((2 ** 61, 1.5)), 1.5)