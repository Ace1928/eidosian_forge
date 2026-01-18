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
def test_dict_bool(self):
    """
        Exercise bool(dict)
        """

    @njit
    def foo(n):
        d = dictobject.new_dict(int32, float32)
        for i in range(n):
            d[i] = i + 1
        return bool(d)
    self.assertEqual(foo(n=0), False)
    self.assertEqual(foo(n=1), True)
    self.assertEqual(foo(n=2), True)
    self.assertEqual(foo(n=100), True)