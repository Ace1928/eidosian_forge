from collections import namedtuple
import math
from functools import reduce
import numpy as np
import operator
import warnings
from llvmlite import ir
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, cgutils
from numba.core.extending import overload, intrinsic
from numba.core.typeconv import Conversion
from numba.core.errors import (TypingError, LoweringError,
from numba.misc.special import literal_unroll
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.builtins import IndexValue, IndexValueType
from numba.extending import overload, register_jitable
@lower_builtin(types.NumberClass, types.Any)
def number_constructor(context, builder, sig, args):
    """
    Call a number class, e.g. np.int32(...)
    """
    if isinstance(sig.return_type, types.Array):
        dt = sig.return_type.dtype

        def foo(*arg_hack):
            return np.array(arg_hack, dtype=dt)
        res = context.compile_internal(builder, foo, sig, args)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    else:
        [val] = args
        [valty] = sig.args
        return context.cast(builder, val, valty, sig.return_type)