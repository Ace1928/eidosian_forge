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
def test_justification(self):
    for pyfunc, case_name in [(center_usecase, 'center'), (ljust_usecase, 'ljust'), (rjust_usecase, 'rjust')]:
        cfunc = njit(pyfunc)
        with self.assertRaises(TypingError) as raises:
            cfunc(UNICODE_EXAMPLES[0], 1.1)
        self.assertIn('The width must be an Integer', str(raises.exception))
        for s in UNICODE_EXAMPLES:
            for width in range(-3, 20):
                self.assertEqual(pyfunc(s, width), cfunc(s, width), "'%s'.%s(%d)?" % (s, case_name, width))