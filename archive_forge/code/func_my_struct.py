import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@njit
def my_struct(values, counter):
    st = structref.new(my_struct_ty)
    my_struct_init(st, values, counter)
    return st