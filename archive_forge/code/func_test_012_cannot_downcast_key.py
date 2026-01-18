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
def test_012_cannot_downcast_key(self):

    @njit
    def foo():
        d = dictobject.new_dict(int32, float64)
        d[0] = 6.0
        return 1j in d
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))