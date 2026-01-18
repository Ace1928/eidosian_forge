import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
def check_dimensions(array_like, name):
    if isinstance(array_like, types.Array):
        if array_like.ndim > 2:
            raise TypeError('{0} has more than 2 dimensions'.format(name))
    elif isinstance(array_like, types.Sequence):
        if isinstance(array_like.key[0], types.Sequence):
            if isinstance(array_like.key[0].key[0], types.Sequence):
                raise TypeError('{0} has more than 2 dimensions'.format(name))