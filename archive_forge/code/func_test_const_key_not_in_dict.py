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
def test_const_key_not_in_dict(self):

    @njit
    def foo():
        a = {'not_a': 2j, 'c': 'd', 'e': np.zeros(4)}
        return a['a']
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn("Key 'a' is not in dict.", str(raises.exception))