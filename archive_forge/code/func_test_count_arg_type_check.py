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
def test_count_arg_type_check(self):
    cfunc = njit(count_with_start_end_usecase)
    with self.assertRaises(TypingError) as raises:
        cfunc('ascii', 'c', 1, 0.5)
    self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('ascii', 'c', 1.2, 7)
    self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('ascii', 12, 1, 7)
    self.assertIn('The substring must be a UnicodeType, not', str(raises.exception))