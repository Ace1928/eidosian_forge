from numba import njit
from functools import reduce
import unittest
def test_basic_filter_external_func(self):
    func = njit(lambda x: x > 0)

    def impl():
        return [y for y in filter(func, range(-10, 10))]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())