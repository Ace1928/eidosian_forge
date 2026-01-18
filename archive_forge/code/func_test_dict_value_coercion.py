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
def test_dict_value_coercion(self):
    p = {(np.int32, np.int32): types.DictType, (np.int32, np.int8): types.DictType, (np.complex128, np.int32): types.DictType, (np.int32, np.complex128): types.LiteralStrKeyDict, (np.int32, np.array): types.LiteralStrKeyDict, (np.array, np.int32): types.LiteralStrKeyDict, (np.int8, np.int32): types.LiteralStrKeyDict, (np.int64, np.float64): types.LiteralStrKeyDict}

    def bar(x):
        pass
    for dts, container in p.items():

        @overload(bar)
        def ol_bar(x):
            self.assertTrue(isinstance(x, container))

            def impl(x):
                pass
            return impl
        ty1, ty2 = dts

        @njit
        def foo():
            d = {'a': ty1(1), 'b': ty2(2)}
            bar(d)
        foo()