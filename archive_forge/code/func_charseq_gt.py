import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.gt)
def charseq_gt(a, b):
    if not _same_kind(a, b):
        return
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:

        def gt_impl(a, b):
            return b < a
        return gt_impl