import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def caching_test_make(x, y):
    struct = MyStruct(values=x, counter=y)
    return struct