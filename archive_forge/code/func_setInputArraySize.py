import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def setInputArraySize(self, argName, size=None):
    """Decorate function with vector-handling code for a single argument
            
            if OpenGL.ERROR_ON_COPY is False, then we return the 
            named argument, converting to the passed array type,
            optionally checking that the array matches size.
            
            if OpenGL.ERROR_ON_COPY is True, then we will dramatically 
            simplify this function, only wrapping if size is True, i.e.
            only wrapping if we intend to do a size check on the array.
            """
    if size is not None:
        arrayType = self.typeOfArg(argName)
        if hasattr(arrayType, 'asArray'):
            self.setPyConverter(argName, arrayhelpers.asArrayTypeSize(arrayType, size))
            self.setCConverter(argName, converters.getPyArgsName(argName))
    return self