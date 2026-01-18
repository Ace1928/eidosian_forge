import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def nullFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, error_checker=None, force_extension=False):
    """Construct a "null" function pointer"""
    if deprecated:
        base = _DeprecatedFunctionPointer
    else:
        base = _NullFunctionPointer
    cls = type(functionName, (base,), {'__doc__': doc, 'deprecated': deprecated})
    if MODULE_ANNOTATIONS:
        if not module:
            module = _find_module()
        if module:
            cls.__module__ = module
    return cls(functionName, dll, resultType, argTypes, argNames, extension=extension, doc=doc, error_checker=error_checker, force_extension=force_extension)