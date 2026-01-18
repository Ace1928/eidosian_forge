import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
def returnPointer(result, baseOperation, pyArgs, cArgs):
    """Return the converted object as result of function
        
        Note: this is a hack that always returns pyArgs[0]!
        """
    return pyArgs[0]