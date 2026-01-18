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
def setCResolver(self, argName, function=NULL):
    """Set C-argument converter for a given argument"""
    if not hasattr(self, 'cResolvers'):
        self.cResolvers = [None] * len(self.wrappedOperation.argNames)
    try:
        if not isinstance(self.wrappedOperation.argNames, list):
            self.wrappedOperation.argNames = list(self.wrappedOperation.argNames)
        i = asList(self.wrappedOperation.argNames).index(argName)
    except ValueError:
        raise AttributeError('No argument named %r left in cConverters: %s' % (argName, self.wrappedOperation.argNames))
    if function is NULL:
        del self.cResolvers[i]
    else:
        self.cResolvers[i] = function
    return self