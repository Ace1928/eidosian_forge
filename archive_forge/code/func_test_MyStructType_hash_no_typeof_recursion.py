import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MyStructType_hash_no_typeof_recursion(self):
    st = MyStruct(1, 2)
    typeof(st)
    self.assertEqual(hash(st), 3)