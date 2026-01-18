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
def test_dict_of_dict_npm(self):
    inner_dict_ty = types.DictType(types.intp, types.intp)

    @njit
    def inner_numba_dict():
        d = Dict.empty(key_type=types.intp, value_type=types.intp)
        return d

    @njit
    def foo(count):
        d = Dict.empty(key_type=types.intp, value_type=inner_dict_ty)
        for i in range(count):
            d[i] = inner_numba_dict()
            for j in range(i + 1):
                d[i][j] = j
        return d
    d = foo(100)
    ct = 0
    for k, dd in d.items():
        ct += 1
        self.assertEqual(len(dd), k + 1)
        for kk, vv in dd.items():
            self.assertEqual(kk, vv)
    self.assertEqual(ct, 100)