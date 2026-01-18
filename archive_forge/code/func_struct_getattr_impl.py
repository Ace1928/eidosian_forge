from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
@lower_getattr_generic(struct_typeclass)
def struct_getattr_impl(context, builder, typ, val, attr):
    utils = _Utils(context, builder, typ)
    dataval = utils.get_data_struct(val)
    ret = getattr(dataval, attr)
    fieldtype = typ.field_dict[attr]
    return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)