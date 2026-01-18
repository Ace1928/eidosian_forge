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
@unittest.skipUnless(numpy_version < (1, 24), 'Setting an array element with a sequence is removed in NumPy 1.24')
def test_comp_nest_with_dependency(self):

    def comp_nest_with_dependency(n):
        l = np.array([[i * j for j in range(i + 1)] for i in range(n)])
        return l
    with self.assertRaises(TypingError) as raises:
        self.check(comp_nest_with_dependency, 5)
    self.assertIn(_header_lead, str(raises.exception))
    self.assertIn('array(undefined,', str(raises.exception))