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
def test_join_non_string_exception(self):
    pyfunc = join_usecase
    cfunc = njit(pyfunc)
    with self.assertRaises(TypingError) as raises:
        cfunc('', [1, 2, 3])
    exc_message = str(raises.exception)
    self.assertIn('During: resolving callee type: BoundFunction', exc_message)
    self.assertIn('reflected list(int', exc_message)