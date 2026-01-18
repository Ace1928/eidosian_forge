import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.contains)
def charseq_contains(a, b):
    if not _same_kind(a, b):
        return
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:
        if _is_bytes(a):

            def contains_impl(a, b):
                return b._to_str() in a._to_str()
        else:

            def contains_impl(a, b):
                return str(b) in str(a)
        return contains_impl