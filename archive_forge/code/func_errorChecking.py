import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def errorChecking(self, func, dll, error_checker=None):
    """Add error checking to the function if appropriate"""
    from OpenGL import error
    if error_checker and _configflags.ERROR_CHECKING:
        func.errcheck = error_checker.glCheckError
    return func