import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.lt)
def charseq_lt(a, b):
    if not _same_kind(a, b):
        return
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:

        def lt_impl(a, b):
            na = len(a)
            nb = len(b)
            n = min(na, nb)
            for i in range(n):
                ca, cb = (left_code(a, i), right_code(b, i))
                if ca != cb:
                    return ca < cb
            return na < nb
        return lt_impl