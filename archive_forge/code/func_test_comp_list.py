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
def test_comp_list(self):
    pyfunc = comp_list
    cfunc = njit((types.intp,))(pyfunc)
    self.assertEqual(cfunc(5), pyfunc(5))
    self.assertEqual(cfunc(0), pyfunc(0))
    self.assertEqual(cfunc(-1), pyfunc(-1))