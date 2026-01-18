import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
def arraySizeOfFirstType(typ, default):
    unitSize = typ.unitSize

    def arraySizeOfFirst(pyArgs, index, baseOperation):
        """Return the array size of the first argument"""
        array = pyArgs[0]
        if array is None:
            return default
        else:
            return unitSize(array)
    return arraySizeOfFirst