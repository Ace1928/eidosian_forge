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
def test_dict_getitem(self):
    """
        Exercise dictionary __getitem__
        """

    @njit
    def foo(keys, vals, target):
        d = dictobject.new_dict(int32, float64)
        for k, v in zip(keys, vals):
            d[k] = v
        return d[target]
    keys = [1, 2, 3]
    vals = [0.1, 0.2, 0.3]
    self.assertEqual(foo(keys, vals, 1), 0.1)
    self.assertEqual(foo(keys, vals, 2), 0.2)
    self.assertEqual(foo(keys, vals, 3), 0.3)
    self.assert_no_memory_leak()
    self.disable_leak_check()
    with self.assertRaises(KeyError):
        foo(keys, vals, 0)
    with self.assertRaises(KeyError):
        foo(keys, vals, 4)