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
def test_getitem_return_type(self):
    d = Dict.empty(types.int64, types.int64[:])
    d[1] = np.arange(10, dtype=np.int64)

    @njit
    def foo(d):
        d[1] += 100
        return d[1]
    foo(d)
    retty = foo.nopython_signatures[0].return_type
    self.assertIsInstance(retty, types.Array)
    self.assertNotIsInstance(retty, types.Optional)
    self.assertPreciseEqual(d[1], np.arange(10, dtype=np.int64) + 100)