from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def get_data_struct(self, val):
    """Get a getter/setter helper for accessing a `StructRefPayload`
        """
    context = self.context
    builder = self.builder
    struct_type = self.struct_type
    data_ptr = self.get_data_pointer(val)
    valtype = struct_type.get_data_type()
    dataval = cgutils.create_struct_proxy(valtype)(context, builder, ref=data_ptr)
    return dataval