from numba import njit
from functools import reduce
import unittest
def test_basic_filter_closure(self):

    def impl():
        return [y for y in filter(lambda x: x > 0, range(-10, 10))]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())