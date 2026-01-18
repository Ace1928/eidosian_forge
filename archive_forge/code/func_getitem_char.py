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
def getitem_char(s, idx):
    idx = normalize_str_idx(idx, len(s))
    cp = _get_code_point(s, idx)
    kind = _codepoint_to_kind(cp)
    if kind == s._kind:
        return _get_str_slice_view(s, idx, 1)
    else:
        is_ascii = _codepoint_is_ascii(cp)
        ret = _empty_string(kind, 1, is_ascii)
        _set_code_point(ret, 0, cp)
        return ret