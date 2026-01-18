import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def functionTypeFor(self, dll):
    """Given a DLL, determine appropriate function type..."""
    if hasattr(dll, 'FunctionType'):
        return dll.FunctionType
    else:
        return self.DEFAULT_FUNCTION_TYPE