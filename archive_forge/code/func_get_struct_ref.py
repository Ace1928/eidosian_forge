from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def get_struct_ref(self, val):
    """Return a helper for accessing a StructRefType
        """
    context = self.context
    builder = self.builder
    struct_type = self.struct_type
    return cgutils.create_struct_proxy(struct_type)(context, builder, value=val)