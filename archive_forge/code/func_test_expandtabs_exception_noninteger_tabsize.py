from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def test_expandtabs_exception_noninteger_tabsize(self):
    pyfunc = expandtabs_with_tabsize_usecase
    cfunc = njit(pyfunc)
    accepted_types = (types.Integer, int)
    with self.assertRaises(TypingError) as raises:
        cfunc('\t', 2.4)
    msg = '"tabsize" must be {}, not float'.format(accepted_types)
    self.assertIn(msg, str(raises.exception))