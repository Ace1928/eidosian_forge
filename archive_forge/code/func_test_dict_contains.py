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
def test_dict_contains(self):
    """
        Exercise operator.contains
        """

    @njit
    def foo(keys, vals, checklist):
        d = dictobject.new_dict(int32, float64)
        for k, v in zip(keys, vals):
            d[k] = v
        out = []
        for k in checklist:
            out.append(k in d)
        return out
    keys = [1, 2, 3]
    vals = [0.1, 0.2, 0.3]
    self.assertEqual(foo(keys, vals, [2, 3, 4, 1, 0]), [True, True, False, True, False])