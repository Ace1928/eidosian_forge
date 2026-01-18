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
def test_list_and_array_as_value(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): types.List(types.intp, initial_value=[1, 2, 3]), types.literal('c'): typeof(np.zeros(5))})

        def impl(x):
            pass
        return impl

    @njit
    def foo():
        b = [1, 2, 3]
        ld = {'a': 1, 'b': b, 'c': np.zeros(5)}
        bar(ld)
    foo()