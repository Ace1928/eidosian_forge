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
@lower_builtin(min, types.BaseTuple)
def min_iterable(context, builder, sig, args):
    argtys = list(sig.args[0])
    args = cgutils.unpack_tuple(builder, args[0])
    return do_minmax(context, builder, argtys, args, operator.lt)