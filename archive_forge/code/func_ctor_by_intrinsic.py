import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@njit
def ctor_by_intrinsic(vs, ctr):
    st = my_struct(vs, counter=ctr)
    st.values += st.values
    st.counter *= ctr
    return st