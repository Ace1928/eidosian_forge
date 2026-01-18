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
@lower_cast(types.DeferredType, types.Any)
@lower_cast(types.DeferredType, types.Boolean)
@lower_cast(types.DeferredType, types.Optional)
def deferred_to_any(context, builder, fromty, toty, val):
    model = context.data_model_manager[fromty]
    val = model.get(builder, val)
    return context.cast(builder, val, fromty.get(), toty)