import ctypes, ctypes.util, operator, sys
from . import model
def new_union_type(self, name):
    return self._new_struct_or_union('union', name, ctypes.Union)