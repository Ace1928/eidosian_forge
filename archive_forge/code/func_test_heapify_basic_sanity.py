import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heapify_basic_sanity(self):
    pyfunc = heapify
    cfunc = jit(nopython=True)(pyfunc)
    a = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    b = self.listimpl(a)
    pyfunc(a)
    cfunc(b)
    self.assertPreciseEqual(a, list(b))
    element_pool = [3.142, -10.0, 5.5, np.nan, -np.inf, np.inf]
    for x in itertools.combinations_with_replacement(element_pool, 6):
        a = list(x)
        b = self.listimpl(a)
        pyfunc(a)
        cfunc(b)
        self.assertPreciseEqual(a, list(b))
    for i in range(len(element_pool)):
        a = [element_pool[i]]
        b = self.listimpl(a)
        pyfunc(a)
        cfunc(b)
        self.assertPreciseEqual(a, list(b))
    a = [(3, 33), (1, 11), (2, 22)]
    b = self.listimpl(a)
    pyfunc(a)
    cfunc(b)
    self.assertPreciseEqual(a, list(b))