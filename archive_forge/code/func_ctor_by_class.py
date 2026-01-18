import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@njit
def ctor_by_class(vs, ctr):
    return MyStruct(values=vs, counter=ctr)