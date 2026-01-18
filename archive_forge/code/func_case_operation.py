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
def case_operation(ascii_func, unicode_func):
    """Generate common case operation performer."""

    def impl(data):
        length = len(data)
        if length == 0:
            return _empty_string(data._kind, length, data._is_ascii)
        if data._is_ascii:
            res = _empty_string(data._kind, length, 1)
            ascii_func(data, res)
            return res
        tmp = _empty_string(PY_UNICODE_4BYTE_KIND, 3 * length, data._is_ascii)
        maxchars = [0]
        newlength = unicode_func(data, length, tmp, maxchars)
        maxchar = maxchars[0]
        newkind = _codepoint_to_kind(maxchar)
        res = _empty_string(newkind, newlength, _codepoint_is_ascii(maxchar))
        for i in range(newlength):
            _set_code_point(res, i, _get_code_point(tmp, i))
        return res
    return impl