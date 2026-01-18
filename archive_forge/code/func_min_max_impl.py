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
@register_jitable
def min_max_impl(iterable, op):
    if isinstance(iterable, types.IterableType):

        def impl(iterable):
            it = iter(iterable)
            return_val = next(it)
            for val in it:
                if op(val, return_val):
                    return_val = val
            return return_val
        return impl