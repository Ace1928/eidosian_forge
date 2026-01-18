from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def get_member_fe_type(self, name):
    """
        StructModel-specific: get the Numba type of the field named *name*.
        """
    pos = self.get_field_position(name)
    return self._members[pos]