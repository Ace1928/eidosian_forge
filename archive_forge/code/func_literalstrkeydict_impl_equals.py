import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
@lower_builtin(operator.eq, types.LiteralStrKeyDict, types.LiteralStrKeyDict)
def literalstrkeydict_impl_equals(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    pred = tu.literal_value == tv.literal_value
    res = context.get_constant(types.boolean, pred)
    return impl_ret_untracked(context, builder, sig.return_type, res)