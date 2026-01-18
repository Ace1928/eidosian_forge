import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(len)
def charseq_len(s):
    if isinstance(s, (types.CharSeq, types.UnicodeCharSeq)):
        get_code = _get_code_impl(s)
        n = s.count
        if n == 0:

            def len_impl(s):
                return 0
            return len_impl
        else:

            def len_impl(s):
                i = n
                code = 0
                while code == 0:
                    i = i - 1
                    if i < 0:
                        break
                    code = get_code(s, i)
                return i + 1
            return len_impl