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
def gen_isAlX(ascii_func, unicode_func):

    def unicode_isAlX(data):

        def impl(data):
            length = len(data)
            if length == 0:
                return False
            if length == 1:
                code_point = _get_code_point(data, 0)
                if data._is_ascii:
                    return ascii_func(code_point)
                else:
                    return unicode_func(code_point)
            if data._is_ascii:
                for i in range(length):
                    code_point = _get_code_point(data, i)
                    if not ascii_func(code_point):
                        return False
            for i in range(length):
                code_point = _get_code_point(data, i)
                if not unicode_func(code_point):
                    return False
            return True
        return impl
    return unicode_isAlX