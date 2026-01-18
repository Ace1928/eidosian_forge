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
@overload_method(types.Integer, '__hash__')
@overload_method(types.Boolean, '__hash__')
def int_hash(val):
    _HASH_I64_MIN = -2 if sys.maxsize <= 2 ** 32 else -4
    _SIGNED_MIN = types.int64(-9223372036854775808)
    _BIG = types.int64 if getattr(val, 'signed', False) else types.uint64

    def impl(val):
        val = _BIG(val)
        mag = abs(val)
        if mag < _PyHASH_MODULUS:
            if val == 0:
                ret = 0
            elif val == _SIGNED_MIN:
                ret = _Py_hash_t(_HASH_I64_MIN)
            else:
                ret = _Py_hash_t(val)
        else:
            needs_negate = False
            if val < 0:
                val = -val
                needs_negate = True
            ret = _long_impl(val)
            if needs_negate:
                ret = -ret
        return process_return(ret)
    return impl