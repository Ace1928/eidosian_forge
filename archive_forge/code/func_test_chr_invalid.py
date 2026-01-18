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
def test_chr_invalid(self):
    pyfunc = chr_usecase
    cfunc = njit(pyfunc)
    for func in (pyfunc, cfunc):
        for v in (-2, _MAX_UNICODE + 1):
            with self.assertRaises(ValueError) as raises:
                func(v)
            self.assertIn('chr() arg not in range', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('abc')
    self.assertIn(_header_lead, str(raises.exception))