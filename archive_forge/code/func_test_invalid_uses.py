import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_invalid_uses(self):
    with self.assertRaisesRegex(ValueError, 'cannot register'):
        structref.register(types.StructRef)
    with self.assertRaisesRegex(ValueError, 'cannot register'):
        structref.define_boxing(types.StructRef, MyStruct)