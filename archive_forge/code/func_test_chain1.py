from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_chain1(self):
    from numba.tests.chained_assign_usecases import chain1
    args = [[np.arange(2)], [np.arange(4, dtype=np.double)]]
    self._test_template(chain1, args)