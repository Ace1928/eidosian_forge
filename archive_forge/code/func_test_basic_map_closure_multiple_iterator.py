from numba import njit
from functools import reduce
import unittest
def test_basic_map_closure_multiple_iterator(self):

    def impl():
        args = (range(10), range(10, 20))
        return [y for y in map(lambda a, b: (a + 10, b + 5), *args)]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())