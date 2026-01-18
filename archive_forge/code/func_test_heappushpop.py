import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heappushpop(self):
    pyfunc = heappushpop
    cfunc = jit(nopython=True)(pyfunc)
    h = self.listimpl([1.0])
    x = cfunc(h, 10.0)
    self.assertPreciseEqual((list(h), x), ([10.0], 1.0))
    self.assertPreciseEqual(type(h[0]), float)
    self.assertPreciseEqual(type(x), float)
    h = self.listimpl([10])
    x = cfunc(h, 9)
    self.assertPreciseEqual((list(h), x), ([10], 9))
    h = self.listimpl([10])
    x = cfunc(h, 11)
    self.assertPreciseEqual((list(h), x), ([11], 10))