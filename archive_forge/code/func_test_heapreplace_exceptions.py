import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heapreplace_exceptions(self):
    pyfunc = heapreplace
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertTypingError() as e:
        cfunc((1, 5, 4), -1)
    msg = 'heap argument must be a list'
    self.assertIn(msg, str(e.exception))
    with self.assertTypingError() as e:
        cfunc(self.listimpl([1, 5, 4]), -1.0)
    msg = 'heap type must be the same as item type'
    self.assertIn(msg, str(e.exception))