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
def calculate_cArgs(pyArgs):
    for index, converter, canCall in cConverters_mapped:
        if canCall:
            try:
                yield converter(pyArgs, index, self)
            except Exception as err:
                if hasattr(err, 'args'):
                    err.args += ('Failure in cConverter %r' % converter, pyArgs, index, self)
                raise
        else:
            yield converter