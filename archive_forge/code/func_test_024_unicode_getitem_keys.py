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
def test_024_unicode_getitem_keys(self):

    @njit
    def foo():
        s = 'aሴ'
        d = {s[0]: 1}
        return d['a']
    self.assertEqual(foo(), foo.py_func())

    @njit
    def foo():
        s = 'abcሴ'
        d = {s[:1]: 1}
        return d['a']
    self.assertEqual(foo(), foo.py_func())