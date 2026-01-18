from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
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