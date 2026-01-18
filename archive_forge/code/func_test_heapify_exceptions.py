import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heapify_exceptions(self):
    pyfunc = heapify
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertTypingError() as e:
        cfunc((1, 5, 4))
    msg = 'heap argument must be a list'
    self.assertIn(msg, str(e.exception))
    with self.assertTypingError() as e:
        cfunc(self.listimpl([1 + 1j, 2 - 3j]))
    msg = "'<' not supported between instances of 'complex' and 'complex'"
    self.assertIn(msg, str(e.exception))