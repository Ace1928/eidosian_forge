import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_getattr_generic(types.EnumClass)
def enum_class_getattr(context, builder, ty, val, attr):
    """
    Return an enum member by attribute name.
    """
    member = getattr(ty.instance_class, attr)
    return context.get_constant_generic(builder, ty.dtype, member.value)