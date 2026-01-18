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
def generate_splitlines_func(is_line_break_func):
    """Generate splitlines performer based on ascii or unicode line breaks."""

    def impl(data, keepends):
        length = len(data)
        result = []
        i = j = 0
        while i < length:
            while i < length:
                code_point = _get_code_point(data, i)
                if is_line_break_func(code_point):
                    break
                i += 1
            eol = i
            if i < length:
                if i + 1 < length:
                    cur_cp = _get_code_point(data, i)
                    next_cp = _get_code_point(data, i + 1)
                    if _Py_ISCARRIAGERETURN(cur_cp) and _Py_ISLINEFEED(next_cp):
                        i += 1
                i += 1
                if keepends:
                    eol = i
            result.append(data[j:eol])
            j = i
        return result
    return impl