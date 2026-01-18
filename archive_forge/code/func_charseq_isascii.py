import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'isascii')
@overload_method(types.CharSeq, 'isascii')
@overload_method(types.Bytes, 'isascii')
def charseq_isascii(s):
    get_code = _get_code_impl(s)

    def impl(s):
        for i in range(len(s)):
            if get_code(s, i) > 127:
                return False
        return True
    return impl