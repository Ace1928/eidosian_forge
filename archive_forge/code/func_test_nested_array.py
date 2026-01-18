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
def test_nested_array(self):

    def nested_array(n):
        l = np.array([np.array([x for x in range(n)]) for y in range(n)])
        return l
    self.check(nested_array, 10)