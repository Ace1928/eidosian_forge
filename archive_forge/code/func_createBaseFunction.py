import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def createBaseFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, error_checker=None):
    """Create a base function for given name
        
        Normally you can just use the dll.name hook to get the object,
        but we want to be able to create different bindings for the 
        same function, so we do the work manually here to produce a
        base function from a DLL.
        """
    from OpenGL import wrapper
    result = None
    try:
        if _configflags.FORWARD_COMPATIBLE_ONLY and dll is self.GL and deprecated:
            result = self.nullFunction(functionName, dll=dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, deprecated=deprecated, error_checker=error_checker)
        else:
            result = self.constructFunction(functionName, dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, error_checker=error_checker)
    except AttributeError as err:
        result = self.nullFunction(functionName, dll=dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, error_checker=error_checker)
    if MODULE_ANNOTATIONS:
        if not module:
            module = _find_module()
        if module:
            result.__module__ = module
    return result