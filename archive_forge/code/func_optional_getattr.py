import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
@lower_getattr_generic(types.Optional)
def optional_getattr(context, builder, typ, value, attr):
    """
    Optional.__getattr__ => redirect to the wrapped type.
    """
    inner_type = typ.type
    val = context.cast(builder, value, typ, inner_type)
    imp = context.get_getattr(inner_type, attr)
    return imp(context, builder, inner_type, val, attr)