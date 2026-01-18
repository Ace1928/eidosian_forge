import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def is_fortran(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of F layout?

    Note: The code is usable as a numba-compiled function
    """
    nd = len(dims)
    firstax = 0
    while firstax < nd and dims[firstax] <= 1:
        firstax += 1
    if firstax >= nd:
        return True
    if itemsize != strides[firstax]:
        return False
    lastax = nd - 1
    while lastax > firstax and dims[lastax] <= 1:
        lastax -= 1
    ax = firstax
    while ax < lastax:
        if strides[ax] * dims[ax] != strides[ax + 1]:
            return False
        ax += 1
    return True