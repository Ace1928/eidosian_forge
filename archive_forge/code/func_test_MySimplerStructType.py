import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MySimplerStructType(self):
    vs = np.arange(10, dtype=np.intp)
    ctr = 13
    first_expected = vs + vs
    first_got = ctor_by_intrinsic(vs, ctr)
    self.assertNotIsInstance(first_got, MyStruct)
    self.assertPreciseEqual(first_expected, get_values(first_got))
    second_expected = first_expected + ctr * ctr
    second_got = compute_fields(first_got)
    self.assertPreciseEqual(second_expected, second_got)