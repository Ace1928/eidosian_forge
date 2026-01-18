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
def test_lower(self):
    pyfunc = lower_usecase
    cfunc = njit(pyfunc)
    extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
    cpython = ['𐐁', '𐐧', '𐑎', '👯', '𐐧𐐧', '𐐧𐑏', 'X𐐧x𐑏', 'İ']
    sigma = ['Σ', 'ͅΣ', 'AͅΣ', 'AͅΣa', 'Σͅ ', '\U0008fffe', 'ⅷ']
    extra_sigma = 'AΣ\u03a2'
    sigma.append(extra_sigma)
    msg = 'Results of "{}".lower() must be equal'
    for s in UNICODE_EXAMPLES + [''] + extras + cpython + sigma:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))