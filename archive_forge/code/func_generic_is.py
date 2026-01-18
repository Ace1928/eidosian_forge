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
@lower_builtin(operator.is_, types.Any, types.Any)
def generic_is(context, builder, sig, args):
    """
    Default implementation for `x is y`
    """
    lhs_type, rhs_type = sig.args
    if lhs_type == rhs_type:
        if lhs_type.mutable:
            msg = 'no default `is` implementation'
            raise LoweringError(msg)
        else:
            try:
                eq_impl = context.get_function(operator.eq, sig)
            except NotImplementedError:
                return cgutils.false_bit
            else:
                return eq_impl(builder, args)
    else:
        return cgutils.false_bit