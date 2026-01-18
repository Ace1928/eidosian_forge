import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(bytes)
def charseq_bytes(s):
    if isinstance(s, types.CharSeq):
        return lambda s: s