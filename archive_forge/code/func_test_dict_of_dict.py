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
def test_dict_of_dict(self):

    @njit
    def foo(k1, k2, v):
        d = Dict()
        z1 = Dict()
        z1[k1 + 1] = v + k1
        z2 = Dict()
        z2[k2 + 2] = v + k2
        d[k1] = z1
        d[k2] = z2
        return d
    k1, k2, v = (100, 200, 321)
    d = foo(k1, k2, v)
    self.assertEqual(dict(d), {k1: {k1 + 1: k1 + v}, k2: {k2 + 2: k2 + v}})