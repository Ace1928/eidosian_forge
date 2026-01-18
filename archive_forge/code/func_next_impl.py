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
@lower_builtin(next, types.IteratorType)
def next_impl(context, builder, sig, args):
    iterty, = sig.args
    iterval, = args
    res = call_iternext(context, builder, iterty, iterval)
    with builder.if_then(builder.not_(res.is_valid()), likely=False):
        context.call_conv.return_user_exc(builder, StopIteration, ())
    return res.yielded_value()