import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@register_jitable
def join_list(sep, parts):
    parts_len = len(parts)
    if parts_len == 0:
        return ''
    sep_len = len(sep)
    length = (parts_len - 1) * sep_len
    kind = sep._kind
    is_ascii = sep._is_ascii
    for p in parts:
        length += len(p)
        kind = _pick_kind(kind, p._kind)
        is_ascii = _pick_ascii(is_ascii, p._is_ascii)
    result = _empty_string(kind, length, is_ascii)
    part = parts[0]
    _strncpy(result, 0, part, 0, len(part))
    dst_offset = len(part)
    for idx in range(1, parts_len):
        _strncpy(result, dst_offset, sep, 0, sep_len)
        dst_offset += sep_len
        part = parts[idx]
        _strncpy(result, dst_offset, part, 0, len(part))
        dst_offset += len(part)
    return result