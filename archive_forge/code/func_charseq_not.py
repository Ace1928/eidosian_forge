import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.not_)
def charseq_not(a):
    if isinstance(a, (types.UnicodeCharSeq, types.CharSeq, types.Bytes)):

        def impl(a):
            return len(a) == 0
        return impl