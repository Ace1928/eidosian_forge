import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def heapiter(self, heap):
    try:
        while 1:
            yield heappop(heap)
    except IndexError:
        pass