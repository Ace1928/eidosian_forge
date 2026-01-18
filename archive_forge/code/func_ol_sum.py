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
@overload(sum)
def ol_sum(iterable, start=0):
    error = None
    if isinstance(start, types.UnicodeType):
        error = ('strings', '')
    elif isinstance(start, types.Bytes):
        error = ('bytes', 'b')
    elif isinstance(start, types.ByteArray):
        error = ('bytearray', 'b')
    if error is not None:
        msg = "sum() can't sum {} [use {}''.join(seq) instead]".format(*error)
        raise TypingError(msg)
    if isinstance(iterable, (types.containers._HomogeneousTuple, types.List, types.ListType, types.Array, types.RangeType)):
        iterator = iter
    elif isinstance(iterable, types.containers._HeterogeneousTuple):
        iterator = literal_unroll
    else:
        return None

    def impl(iterable, start=0):
        acc = start
        for x in iterator(iterable):
            acc = acc + x
        return acc
    return impl