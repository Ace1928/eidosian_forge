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
def expandtabs_impl(data, tabsize=8):
    length = len(data)
    j = line_pos = 0
    found = False
    for i in range(length):
        code_point = _get_code_point(data, i)
        if code_point == _Py_TAB:
            found = True
            if tabsize > 0:
                incr = tabsize - line_pos % tabsize
                if j > sys.maxsize - incr:
                    raise OverflowError('new string is too long')
                line_pos += incr
                j += incr
        else:
            if j > sys.maxsize - 1:
                raise OverflowError('new string is too long')
            line_pos += 1
            j += 1
            if code_point in (_Py_LINEFEED, _Py_CARRIAGE_RETURN):
                line_pos = 0
    if not found:
        return data
    res = _empty_string(data._kind, j, data._is_ascii)
    j = line_pos = 0
    for i in range(length):
        code_point = _get_code_point(data, i)
        if code_point == _Py_TAB:
            if tabsize > 0:
                incr = tabsize - line_pos % tabsize
                line_pos += incr
                for idx in range(j, j + incr):
                    _set_code_point(res, idx, _Py_SPACE)
                j += incr
        else:
            line_pos += 1
            _set_code_point(res, j, code_point)
            j += 1
            if code_point in (_Py_LINEFEED, _Py_CARRIAGE_RETURN):
                line_pos = 0
    return res