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
@lower_builtin('getiter', types.DictType)
def impl_dict_getiter(context, builder, sig, args):
    """Implement iter(Dict).  Semantically equivalent to dict.keys()
    """
    [td] = sig.args
    [d] = args
    iterablety = types.DictKeysIterableType(td)
    it = context.make_helper(builder, iterablety.iterator_type)
    fnty = ir.FunctionType(ir.VoidType(), [ll_dictiter_type, ll_dict_type])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_iter')
    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())
    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)
    it.parent = d
    dp = _container_get_data(context, builder, iterablety.parent, args[0])
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(context, builder, sig.return_type, it._getvalue())