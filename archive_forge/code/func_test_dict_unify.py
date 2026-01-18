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
def test_dict_unify(self):

    @njit
    def foo(x):
        if x + 7 > 4:
            a = {'a': 2j, 'c': 'd', 'e': np.zeros(4)}
        else:
            a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
        return a['c']
    self.assertEqual(foo(100), 'd')
    self.assertEqual(foo(-100), 'CAT')
    self.assertEqual(foo(100), foo.py_func(100))
    self.assertEqual(foo(-100), foo.py_func(-100))