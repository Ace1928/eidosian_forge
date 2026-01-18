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
@lower_builtin('iternext', types.DictIteratorType)
@iternext_impl(RefType.BORROWED)
def impl_iterator_iternext(context, builder, sig, args, result):
    iter_type = sig.args[0]
    it = context.make_helper(builder, iter_type, args[0])
    p2p_bytes = ll_bytes.as_pointer()
    iternext_fnty = ir.FunctionType(ll_status, [ll_bytes, p2p_bytes, p2p_bytes])
    iternext = cgutils.get_or_insert_function(builder.module, iternext_fnty, 'numba_dict_iter_next')
    key_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    val_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    status = builder.call(iternext, (it.state, key_raw_ptr, val_raw_ptr))
    is_valid = builder.icmp_unsigned('==', status, status.type(0))
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        yield_type = iter_type.yield_type
        key_ty, val_ty = iter_type.parent.keyvalue_type
        dm_key = context.data_model_manager[key_ty]
        dm_val = context.data_model_manager[val_ty]
        key_ptr = builder.bitcast(builder.load(key_raw_ptr), dm_key.get_data_type().as_pointer())
        val_ptr = builder.bitcast(builder.load(val_raw_ptr), dm_val.get_data_type().as_pointer())
        key = dm_key.load_from_data_pointer(builder, key_ptr)
        val = dm_val.load_from_data_pointer(builder, val_ptr)
        if isinstance(iter_type.iterable, DictItemsIterableType):
            tup = context.make_tuple(builder, yield_type, [key, val])
            result.yield_(tup)
        elif isinstance(iter_type.iterable, DictKeysIterableType):
            result.yield_(key)
        elif isinstance(iter_type.iterable, DictValuesIterableType):
            result.yield_(val)
        else:
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))