import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
@ClassBuilder.class_impl_registry.lower_getattr_generic(types.ClassInstanceType)
def get_attr_impl(context, builder, typ, value, attr):
    """
    Generic getattr() for @jitclass instances.
    """
    if attr in typ.struct:
        inst = context.make_helper(builder, typ, value=value)
        data_pointer = inst.data
        data = context.make_data_helper(builder, typ.get_data_type(), ref=data_pointer)
        return imputils.impl_ret_borrowed(context, builder, typ.struct[attr], getattr(data, _mangle_attr(attr)))
    elif attr in typ.jit_props:
        getter = typ.jit_props[attr]['get']
        sig = templates.signature(None, typ)
        dispatcher = types.Dispatcher(getter)
        sig = dispatcher.get_call_type(context.typing_context, [typ], {})
        call = context.get_function(dispatcher, sig)
        out = call(builder, [value])
        _add_linking_libs(context, call)
        return imputils.impl_ret_new_ref(context, builder, sig.return_type, out)
    raise NotImplementedError('attribute {0!r} not implemented'.format(attr))