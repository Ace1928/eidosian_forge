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
def scalar_result_expected(mandatory_input, optional_input):
    opt_is_none = optional_input in (None, types.none)
    if isinstance(mandatory_input, types.Array) and mandatory_input.ndim == 1:
        return opt_is_none
    if isinstance(mandatory_input, types.BaseTuple):
        if all((isinstance(x, (types.Number, types.Boolean)) for x in mandatory_input.types)):
            return opt_is_none
        elif len(mandatory_input.types) == 1 and isinstance(mandatory_input.types[0], types.BaseTuple):
            return opt_is_none
    if isinstance(mandatory_input, (types.Number, types.Boolean)):
        return opt_is_none
    if isinstance(mandatory_input, types.Sequence):
        if not isinstance(mandatory_input.key[0], types.Sequence) and opt_is_none:
            return True
    return False