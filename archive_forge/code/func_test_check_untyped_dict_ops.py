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
def test_check_untyped_dict_ops(self):
    d = Dict()
    self.assertFalse(d._typed)
    self.assertEqual(len(d), 0)
    self.assertEqual(str(d), str({}))
    self.assertEqual(list(iter(d)), [])
    with self.assertRaises(KeyError) as raises:
        d[1]
    self.assertEqual(str(raises.exception), str(KeyError(1)))
    with self.assertRaises(KeyError) as raises:
        del d[1]
    self.assertEqual(str(raises.exception), str(KeyError(1)))
    with self.assertRaises(KeyError):
        d.pop(1)
    self.assertEqual(str(raises.exception), str(KeyError(1)))
    self.assertIs(d.pop(1, None), None)
    self.assertIs(d.get(1), None)
    with self.assertRaises(KeyError) as raises:
        d.popitem()
    self.assertEqual(str(raises.exception), str(KeyError('dictionary is empty')))
    with self.assertRaises(TypeError) as raises:
        d.setdefault(1)
    self.assertEqual(str(raises.exception), str(TypeError('invalid operation on untyped dictionary')))
    self.assertFalse(1 in d)
    self.assertFalse(d._typed)