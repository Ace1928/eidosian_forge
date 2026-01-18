from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
@classmethod
def get_implementation(cls, *args):
    if cls.CHOSEN is None:
        for possible in cls.IMPLEMENTATION_CLASSES:
            implementation = possible()
            if possible:
                Implementation.CHOSEN = implementation
                break
    return cls.CHOSEN