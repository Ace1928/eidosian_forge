import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_issue_1469_3(self):
    nptype = np.dtype([('a', np.float64, (4,))])
    nbtype = types.Array(numpy_support.from_dtype(nptype), 2, 'C')
    natype = types.NestedArray(types.float64, (4,))
    fields = [('a', {'type': natype, 'offset': 0})]
    rectype = types.Record(fields=fields, size=32, aligned=False)
    expected = types.Array(rectype, 2, 'C', aligned=False)
    self.assertEqual(nbtype, expected)