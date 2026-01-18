import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@register_jitable
def binary_search_with_guess(key, arr, length, guess):
    imin = 0
    imax = length
    if key > arr[length - 1]:
        return length
    elif key < arr[0]:
        return -1
    if length <= 4:
        i = 1
        while i < length and key >= arr[i]:
            i += 1
        return i - 1
    if guess > length - 3:
        guess = length - 3
    if guess < 1:
        guess = 1
    if key < arr[guess]:
        if key < arr[guess - 1]:
            imax = guess - 1
            if guess > LIKELY_IN_CACHE_SIZE and key >= arr[guess - LIKELY_IN_CACHE_SIZE]:
                imin = guess - LIKELY_IN_CACHE_SIZE
        else:
            return guess - 1
    elif key < arr[guess + 1]:
        return guess
    elif key < arr[guess + 2]:
        return guess + 1
    else:
        imin = guess + 2
        if guess < length - LIKELY_IN_CACHE_SIZE - 1 and key < arr[guess + LIKELY_IN_CACHE_SIZE]:
            imax = guess + LIKELY_IN_CACHE_SIZE
    while imin < imax:
        imid = imin + (imax - imin >> 1)
        if key >= arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin - 1