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
def test_array_comp_inferred_dtype_nested(self):

    def array_comp(n):
        l = np.array([[i * j for j in range(n)] for i in range(n)])
        return l
    self.check(array_comp, 10)