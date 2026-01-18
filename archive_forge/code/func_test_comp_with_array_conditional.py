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
def test_comp_with_array_conditional(self):

    def comp_with_array_conditional(n):
        l = np.array([i for i in range(n) if i % 2 == 1])
        return l
    self.check(comp_with_array_conditional, 10, assert_allocate_list=True)