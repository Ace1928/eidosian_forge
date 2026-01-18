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
def test_comp_nest_with_array(self):

    def comp_nest_with_array(n):
        l = np.array([[i * j for j in range(n)] for i in range(n)])
        return l
    self.check(comp_nest_with_array, 5)
    if PARALLEL_SUPPORTED:
        self.check(comp_nest_with_array, 5, run_parallel=True)