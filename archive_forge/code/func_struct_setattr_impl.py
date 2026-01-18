from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
@lower_setattr_generic(struct_typeclass)
def struct_setattr_impl(context, builder, sig, args, attr):
    [inst_type, val_type] = sig.args
    [instance, val] = args
    utils = _Utils(context, builder, inst_type)
    dataval = utils.get_data_struct(instance)
    field_type = inst_type.field_dict[attr]
    casted = context.cast(builder, val, val_type, field_type)
    old_value = getattr(dataval, attr)
    context.nrt.incref(builder, val_type, casted)
    context.nrt.decref(builder, val_type, old_value)
    setattr(dataval, attr, casted)