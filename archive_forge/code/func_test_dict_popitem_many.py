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
def test_dict_popitem_many(self):
    """
        Exercise dictionary .popitem
        """

    @njit
    def core(d, npop):
        keysum, valsum = (0, 0)
        for _ in range(npop):
            k, v = d.popitem()
            keysum += k
            valsum -= v
        return (keysum, valsum)

    @njit
    def foo(keys, vals, npop):
        d = dictobject.new_dict(int32, int32)
        for k, v in zip(keys, vals):
            d[k] = v
        return core(d, npop)
    keys = [1, 2, 3]
    vals = [10, 20, 30]
    for i in range(len(keys)):
        self.assertEqual(foo(keys, vals, npop=3), core.py_func(dict(zip(keys, vals)), npop=3))
    self.assert_no_memory_leak()
    self.disable_leak_check()
    with self.assertRaises(KeyError):
        foo(keys, vals, npop=4)