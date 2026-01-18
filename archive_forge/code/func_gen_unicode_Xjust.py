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
def gen_unicode_Xjust(STRING_FIRST):

    def unicode_Xjust(string, width, fillchar=' '):
        if not isinstance(width, types.Integer):
            raise TypingError('The width must be an Integer')
        if isinstance(fillchar, types.UnicodeCharSeq):
            if STRING_FIRST:

                def ljust_impl(string, width, fillchar=' '):
                    return string.ljust(width, str(fillchar))
                return ljust_impl
            else:

                def rjust_impl(string, width, fillchar=' '):
                    return string.rjust(width, str(fillchar))
                return rjust_impl
        if not (fillchar == ' ' or isinstance(fillchar, (types.Omitted, types.UnicodeType))):
            raise TypingError('The fillchar must be a UnicodeType')

        def impl(string, width, fillchar=' '):
            str_len = len(string)
            fillchar_len = len(fillchar)
            if fillchar_len != 1:
                raise ValueError('The fill character must be exactly one character long')
            if width <= str_len:
                return string
            newstr = fillchar * (width - str_len)
            if STRING_FIRST:
                return string + newstr
            else:
                return newstr + string
        return impl
    return unicode_Xjust