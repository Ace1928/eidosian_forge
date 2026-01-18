import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_getitem_key(self):
    pyfunc = getitem_key
    cfunc = jit(nopython=True)(pyfunc)
    for x, i in [(np.array('123'), ()), (np.array(['123']), 0), (np.array(b'123'), ()), (np.array([b'123']), 0)]:
        d1 = {}
        d2 = Dict.empty(from_dtype(x.dtype), types.int64)
        pyfunc(d1, x, i)
        cfunc(d2, x, i)
        self.assertEqual(d1, d2)
        str(d2)