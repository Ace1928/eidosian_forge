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
def test_mutation_not_carried(self):

    def bar(d):
        ...

    @overload(bar)
    def ol_bar(d):
        if d.initial_value is None:
            return lambda d: literally(d)
        self.assertTrue(isinstance(d, types.DictType))
        self.assertEqual(d.initial_value, {'a': 1, 'b': 2, 'c': 3})
        return lambda d: d

    @njit
    def foo():
        x = {'a': 1, 'b': 2, 'c': 3}
        x['d'] = 4
        bar(x)
    foo()