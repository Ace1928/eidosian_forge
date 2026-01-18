import ctypes
import weakref
from OpenGL._bytes import long, integer_types
def getAsParam(self):
    """Gets as a ctypes pointer to the underlying structure"""
    return ctypes.pointer(self)