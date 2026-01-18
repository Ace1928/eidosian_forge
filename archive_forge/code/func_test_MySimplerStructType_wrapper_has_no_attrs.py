import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MySimplerStructType_wrapper_has_no_attrs(self):
    vs = np.arange(10, dtype=np.intp)
    ctr = 13
    wrapper = ctor_by_intrinsic(vs, ctr)
    self.assertIsInstance(wrapper, structref.StructRefProxy)
    with self.assertRaisesRegex(AttributeError, 'values'):
        wrapper.values
    with self.assertRaisesRegex(AttributeError, 'counter'):
        wrapper.counter