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
def calculate_pyArgs(args):
    if pyConverters_length > len(args):
        raise ValueError('%s requires %r arguments (%s), received %s: %r' % (wrappedOperation.__name__, pyConverters_length, ', '.join(self.pyConverterNames), len(args), args))
    for index, converter, isNone in pyConverters_mapped:
        if isNone:
            yield args[index]
        else:
            try:
                yield converter(args[index], self, args)
            except IndexError as err:
                yield NULL
            except Exception as err:
                if hasattr(err, 'args'):
                    err.args += (converter,)
                raise