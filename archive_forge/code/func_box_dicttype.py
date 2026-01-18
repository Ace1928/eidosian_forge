from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
@box(types.DictType)
def box_dicttype(typ, val, c):
    context = c.context
    builder = c.builder
    ctor = cgutils.create_struct_proxy(typ)
    dstruct = ctor(context, builder, value=val)
    boxed_meminfo = c.box(types.MemInfoPointer(types.voidptr), dstruct.meminfo)
    modname = c.context.insert_const_string(c.builder.module, 'numba.typed.typeddict')
    typeddict_mod = c.pyapi.import_module_noblock(modname)
    fmp_fn = c.pyapi.object_getattr_string(typeddict_mod, '_from_meminfo_ptr')
    dicttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))
    result_var = builder.alloca(c.pyapi.pyobj)
    builder.store(cgutils.get_null_value(c.pyapi.pyobj), result_var)
    with builder.if_then(cgutils.is_not_null(builder, dicttype_obj)):
        res = c.pyapi.call_function_objargs(fmp_fn, (boxed_meminfo, dicttype_obj))
        c.pyapi.decref(fmp_fn)
        c.pyapi.decref(typeddict_mod)
        c.pyapi.decref(boxed_meminfo)
        builder.store(res, result_var)
    return builder.load(result_var)