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
def test_022_references_juggle(self):

    @njit
    def foo():
        d = dictobject.new_dict(int32, float64)
        e = d
        d[1] = 12.0
        e[2] = 14.0
        e = dictobject.new_dict(int32, float64)
        e[1] = 100.0
        e[2] = 1000.0
        f = d
        d = e
        k1 = [x for x in d.items()]
        k2 = [x for x in e.items()]
        k3 = [x for x in f.items()]
        return (k1, k2, k3)
    k1, k2, k3 = foo()
    self.assertEqual(k1, [(1, 100.0), (2, 1000.0)])
    self.assertEqual(k2, [(1, 100.0), (2, 1000.0)])
    self.assertEqual(k3, [(1, 12), (2, 14)])