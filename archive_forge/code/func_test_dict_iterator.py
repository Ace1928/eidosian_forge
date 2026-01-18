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
def test_dict_iterator(self):

    @njit
    def fun1():
        dd = Dict.empty(key_type=types.intp, value_type=types.intp)
        dd[0] = 10
        dd[1] = 20
        dd[2] = 30
        return (list(dd.keys()), list(dd.values()))

    @njit
    def fun2():
        dd = Dict.empty(key_type=types.intp, value_type=types.intp)
        dd[4] = 77
        dd[5] = 88
        dd[6] = 99
        return (list(dd.keys()), list(dd.values()))
    res1 = fun1()
    res2 = fun2()
    self.assertEqual([0, 1, 2], res1[0])
    self.assertEqual([10, 20, 30], res1[1])
    self.assertEqual([4, 5, 6], res2[0])
    self.assertEqual([77, 88, 99], res2[1])