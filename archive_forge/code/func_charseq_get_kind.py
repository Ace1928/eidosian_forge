import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, '_get_kind')
@overload_method(types.CharSeq, '_get_kind')
def charseq_get_kind(s):
    get_code = _get_code_impl(s)

    def impl(s):
        max_code = 0
        for i in range(len(s)):
            code = get_code(s, i)
            if code > max_code:
                max_code = code
        if max_code > 65535:
            return unicode.PY_UNICODE_4BYTE_KIND
        if max_code > 255:
            return unicode.PY_UNICODE_2BYTE_KIND
        return unicode.PY_UNICODE_1BYTE_KIND
    return impl