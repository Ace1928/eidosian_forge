from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def new_struct_ref(self, mi):
    """Encapsulate the MemInfo from a `StructRefPayload` in a `StructRef`
        """
    context = self.context
    builder = self.builder
    struct_type = self.struct_type
    st = cgutils.create_struct_proxy(struct_type)(context, builder)
    st.meminfo = mi
    return st