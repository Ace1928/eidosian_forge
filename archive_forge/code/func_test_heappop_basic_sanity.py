import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heappop_basic_sanity(self):
    pyfunc = heappop
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
        yield [(3, 33), (1, 111), (2, 2222)]
        yield np.full(5, fill_value=np.nan).tolist()
        yield np.linspace(-10, -5, 100).tolist()
    for a in a_variations():
        heapify(a)
        b = self.listimpl(a)
        for i in range(len(a)):
            val_py = pyfunc(a)
            val_c = cfunc(b)
            self.assertPreciseEqual(a, list(b))
            self.assertPreciseEqual(val_py, val_c)