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
def test_dict_setdefault(self):
    """
        Exercise dict.setdefault
        """

    @njit
    def foo():
        d = dictobject.new_dict(int32, float64)
        d.setdefault(1, 1.2)
        a = d.get(1)
        d[1] = 2.3
        b = d.get(1)
        d[2] = 3.4
        d.setdefault(2, 4.5)
        c = d.get(2)
        return (a, b, c)
    self.assertEqual(foo(), (1.2, 2.3, 3.4))