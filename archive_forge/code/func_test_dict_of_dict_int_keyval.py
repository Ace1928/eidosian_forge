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
def test_dict_of_dict_int_keyval(self):

    def inner_numba_dict():
        d = Dict.empty(key_type=types.intp, value_type=types.intp)
        return d
    d = Dict.empty(key_type=types.intp, value_type=types.DictType(types.intp, types.intp))

    def usecase(d, make_inner_dict):
        for i in range(100):
            mid = make_inner_dict()
            for j in range(i + 1):
                mid[j] = j * 10000
            d[i] = mid
        return d
    got = usecase(d, inner_numba_dict)
    expect = usecase({}, dict)
    self.assertIsInstance(expect, dict)
    self.assertEqual(dict(got), expect)
    for where in [12, 3, 6, 8, 10]:
        del got[where]
        del expect[where]
        self.assertEqual(dict(got), expect)