import ctypes, ctypes.util, operator, sys
from . import model
def new_struct_type(self, name):
    return self._new_struct_or_union('struct', name, ctypes.Structure)