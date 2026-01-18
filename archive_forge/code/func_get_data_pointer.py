from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def get_data_pointer(self, val):
    """Get the data pointer to the payload from a `StructRefType`.
        """
    context = self.context
    builder = self.builder
    struct_type = self.struct_type
    structval = self.get_struct_ref(val)
    meminfo = structval.meminfo
    data_ptr = context.nrt.meminfo_data(builder, meminfo)
    valtype = struct_type.get_data_type()
    model = context.data_model_manager[valtype]
    alloc_type = model.get_value_type()
    data_ptr = builder.bitcast(data_ptr, alloc_type.as_pointer())
    return data_ptr