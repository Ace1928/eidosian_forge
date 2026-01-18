import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
@overload(_defer_hash)
def ol_defer_hash(obj, hash_func):
    err_msg = f"unhashable type: '{obj}'"

    def impl(obj, hash_func):
        if hash_func is None:
            raise TypeError(err_msg)
        else:
            return hash_func()
    return impl