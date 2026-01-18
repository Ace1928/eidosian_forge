from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def define_boxing(struct_type, obj_class):
    """Define the boxing & unboxing logic for `struct_type` to `obj_class`.

    Defines both boxing and unboxing.

    - boxing turns an instance of `struct_type` into a PyObject of `obj_class`
    - unboxing turns an instance of `obj_class` into an instance of
      `struct_type` in jit-code.


    Use this directly instead of `define_proxy()` when the user does not
    want any constructor to be defined.
    """
    if struct_type is types.StructRef:
        raise ValueError(f'cannot register {types.StructRef}')
    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_struct_ref(typ, val, c):
        """
        Convert a raw pointer to a Python int.
        """
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.get_struct_ref(val)
        meminfo = struct_ref.meminfo
        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)
        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))
        res = c.pyapi.call_function_objargs(ctor_pyfunc, [ty_pyobj, boxed_meminfo])
        c.pyapi.decref(ctor_pyfunc)
        c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        return res

    @unbox(struct_type)
    def unbox_struct_ref(typ, obj, c):
        mi_obj = c.pyapi.object_getattr_string(obj, '_meminfo')
        mip_type = types.MemInfoPointer(types.voidptr)
        mi = c.unbox(mip_type, mi_obj).value
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.new_struct_ref(mi)
        out = struct_ref._getvalue()
        c.pyapi.decref(mi_obj)
        return NativeValue(out)