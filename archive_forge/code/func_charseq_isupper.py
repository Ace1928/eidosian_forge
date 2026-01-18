import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.CharSeq, 'isupper')
def charseq_isupper(s):

    def impl(s):
        return not not s._to_str().isupper()
    return impl