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
def test_020_string_key(self):

    @njit
    def foo():
        d = dictobject.new_dict(types.unicode_type, float64)
        d['a'] = 1.0
        d['b'] = 2.0
        d['c'] = 3.0
        d['d'] = 4.0
        out = []
        for x in d.items():
            out.append(x)
        return (out, d['a'])
    items, da = foo()
    self.assertEqual(items, [('a', 1.0), ('b', 2.0), ('c', 3.0), ('d', 4)])
    self.assertEqual(da, 1.0)