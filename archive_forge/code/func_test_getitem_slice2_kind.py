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
def test_getitem_slice2_kind(self):
    pyfunc = getitem_check_kind_usecase
    cfunc = njit(pyfunc)
    samples = ['abcሴሴ', '¡¡¡着着着']
    for s in samples:
        for i in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
            for j in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
                sl = slice(i, j)
                self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d]?" % (s, i, j))