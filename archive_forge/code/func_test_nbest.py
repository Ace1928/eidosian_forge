import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_nbest(self):
    cfunc_heapify = jit(nopython=True)(heapify)
    cfunc_heapreplace = jit(nopython=True)(heapreplace)
    data = self.rnd.choice(range(2000), 1000).tolist()
    heap = self.listimpl(data[:10])
    cfunc_heapify(heap)
    for item in data[10:]:
        if item > heap[0]:
            cfunc_heapreplace(heap, item)
    self.assertPreciseEqual(list(self.heapiter(list(heap))), sorted(data)[-10:])