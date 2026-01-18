import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@register_jitable
def charseq_get_value(a, i):
    """Access i-th item of CharSeq object via code value.

    null code is interpreted as IndexError
    """
    code = charseq_get_code(a, i)
    if code == 0:
        raise IndexError('index out of range')
    return code