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
@overload(hash)
def hash_overload(obj):
    attempt_generic_msg = f"No __hash__ is defined for object of type '{obj}' and a generic hash() cannot be performed as there is no suitable object represention in Numba compiled code!"

    def impl(obj):
        if hasattr(obj, '__hash__'):
            return _defer_hash(obj, getattr(obj, '__hash__'))
        else:
            raise TypeError(attempt_generic_msg)
    return impl