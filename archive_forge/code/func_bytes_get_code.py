import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@register_jitable
def bytes_get_code(a, i):
    """Access i-th item of Bytes object.
        """
    return a[i]