import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
def setInputArraySizeType(baseOperation, size, type, argName=0):
    """Decorate function with vector-handling code for a single argument
    
    if OpenGL.ERROR_ON_COPY is False, then we return the 
    named argument, converting to the passed array type,
    optionally checking that the array matches size.
    
    if OpenGL.ERROR_ON_COPY is True, then we will dramatically 
    simplify this function, only wrapping if size is True, i.e.
    only wrapping if we intend to do a size check on the array.
    """
    from OpenGL import wrapper
    return wrapper.wrapper(baseOperation).setInputArraySize(argName, size)