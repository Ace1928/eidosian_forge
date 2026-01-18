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
def test_dict_to_from_meminfo(self):
    """
        Exercise dictobject.{_as_meminfo, _from_meminfo}
        """

    @njit
    def make_content(nelem):
        for i in range(nelem):
            yield (i, i + (i + 1) / 100)

    @njit
    def boxer(nelem):
        d = dictobject.new_dict(int32, float64)
        for k, v in make_content(nelem):
            d[k] = v
        return dictobject._as_meminfo(d)
    dcttype = types.DictType(int32, float64)

    @njit
    def unboxer(mi):
        d = dictobject._from_meminfo(mi, dcttype)
        return list(d.items())
    mi = boxer(10)
    self.assertEqual(mi.refcount, 1)
    got = unboxer(mi)
    expected = list(make_content.py_func(10))
    self.assertEqual(got, expected)