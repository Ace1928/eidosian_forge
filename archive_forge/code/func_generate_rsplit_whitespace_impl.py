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
def generate_rsplit_whitespace_impl(isspace_func):
    """Generate whitespace rsplit func based on either ascii or unicode"""

    def rsplit_whitespace_impl(data, sep=None, maxsplit=-1):
        if maxsplit < 0:
            maxsplit = sys.maxsize
        result = []
        i = len(data) - 1
        while maxsplit > 0:
            while i >= 0:
                code_point = _get_code_point(data, i)
                if not isspace_func(code_point):
                    break
                i -= 1
            if i < 0:
                break
            j = i
            i -= 1
            while i >= 0:
                code_point = _get_code_point(data, i)
                if isspace_func(code_point):
                    break
                i -= 1
            result.append(data[i + 1:j + 1])
            maxsplit -= 1
        if i >= 0:
            while i >= 0:
                code_point = _get_code_point(data, i)
                if not isspace_func(code_point):
                    break
                i -= 1
            if i >= 0:
                result.append(data[0:i + 1])
        return result[::-1]
    return rsplit_whitespace_impl