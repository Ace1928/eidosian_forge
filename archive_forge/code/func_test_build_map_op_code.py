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
def test_build_map_op_code(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):

        def impl(x):
            pass
        return impl

    @njit
    def foo():
        a = {'a': {'b1': 10, 'b2': 'string'}}
        bar(a)
    foo()