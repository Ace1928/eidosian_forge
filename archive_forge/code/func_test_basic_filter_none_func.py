from numba import njit
from functools import reduce
import unittest
def test_basic_filter_none_func(self):

    def impl():
        return [y for y in filter(None, range(-10, 10))]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())