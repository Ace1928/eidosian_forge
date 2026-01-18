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
def test_conflicting_value_type(self):

    @njit
    def foo(k, v, w):
        d = Dict()
        d[k] = v
        d[k] = w
        return d
    k, v, w = (123, 321, 32.1)
    with self.assertRaises(TypingError) as raises:
        foo(k, v, w)
    self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))