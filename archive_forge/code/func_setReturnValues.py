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
def setReturnValues(self, function=NULL):
    """Set the return-of-results function for the whole wrapper"""
    if function is NULL:
        try:
            del self.returnValues
        except Exception:
            pass
    elif hasattr(self, 'returnValues'):
        if isinstance(self.returnValues, MultiReturn):
            self.returnValues.append(function)
        else:
            self.returnValues = MultiReturn(self.returnValues, function)
    else:
        self.returnValues = function
    return self