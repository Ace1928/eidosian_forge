from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def define_attributes(struct_typeclass):
    """Define attributes on `struct_typeclass`.

    Defines both setters and getters in jit-code.

    This is called directly in `register()`.
    """

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, typ, attr):
            if attr in typ.field_dict:
                attrty = typ.field_dict[attr]
                return attrty

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, typ, val, attr):
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

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