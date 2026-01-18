from numba import njit
from functools import reduce
import unittest
def test_basic_map_external_func(self):
    func = njit(lambda x: x + 10)

    def impl():
        return [y for y in map(func, range(10))]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())