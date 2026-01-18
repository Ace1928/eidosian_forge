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
def test_dict_not_unify(self):

    @njit
    def key_mismatch(x):
        if x + 7 > 4:
            a = {'BAD_KEY': 2j, 'c': 'd', 'e': np.zeros(4)}
        else:
            a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return a['a']
    with self.assertRaises(TypingError) as raises:
        key_mismatch(100)
    self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))

    @njit
    def value_type_mismatch(x):
        if x + 7 > 4:
            a = {'a': 2j, 'c': 'd', 'e': np.zeros((4, 3))}
        else:
            a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return a['a']
    with self.assertRaises(TypingError) as raises:
        value_type_mismatch(100)
    self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))