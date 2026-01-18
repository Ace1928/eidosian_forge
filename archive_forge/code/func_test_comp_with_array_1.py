import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def test_comp_with_array_1(self):

    def comp_with_array_1(n):
        m = n * 2
        l = np.array([i + m for i in range(n)])
        return l
    self.check(comp_with_array_1, 5)
    if PARALLEL_SUPPORTED:
        self.check(comp_with_array_1, 5, run_parallel=True)