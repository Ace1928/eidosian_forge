import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
def test_dict_update(self):
    """
        Tests dict.update works with various dictionaries.
        """
    n = 10

    def f1(n):
        """
            Test update with a regular dictionary.
            """
        d1 = {i: i + 1 for i in range(n)}
        d2 = {3 * i: i for i in range(n)}
        d1.update(d2)
        return d1
    py_func = f1
    cfunc = njit()(f1)
    a = py_func(n)
    b = cfunc(n)
    self.assertEqual(a, b)

    def f2(n):
        """
            Test update where one of the dictionaries
            is created as a Python literal.
            """
        d1 = {1: 2, 3: 4, 5: 6}
        d2 = {3 * i: i for i in range(n)}
        d1.update(d2)
        return d1
    py_func = f2
    cfunc = njit()(f2)
    a = py_func(n)
    b = cfunc(n)
    self.assertEqual(a, b)