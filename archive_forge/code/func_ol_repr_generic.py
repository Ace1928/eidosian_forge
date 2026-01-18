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
@overload(repr)
def ol_repr_generic(obj):
    missing_repr_format = f'<object type:{obj}>'

    def impl(obj):
        attr = '__repr__'
        if hasattr(obj, attr) == True:
            return getattr(obj, attr)()
        else:
            return missing_repr_format
    return impl