import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_contains_getitem2(self):
    pyfunc = contains_getitem2
    cfunc = jit(nopython=True)(pyfunc)
    x = np.array('123')
    y = np.array('12345')
    self._test(pyfunc, cfunc, x, (), y, ())
    self._test(pyfunc, cfunc, y, (), x, ())
    x = np.array(b'123')
    y = np.array(b'12345')
    self._test(pyfunc, cfunc, x, (), y, ())
    self._test(pyfunc, cfunc, y, (), x, ())
    x = ('123',)
    y = np.array('12345')
    self._test(pyfunc, cfunc, x, 0, y, ())
    self._test(pyfunc, cfunc, y, (), x, 0)
    x = (b'123',)
    y = np.array(b'12345')
    self._test(pyfunc, cfunc, x, 0, y, ())
    self._test(pyfunc, cfunc, y, (), x, 0)