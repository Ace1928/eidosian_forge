from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def asIntConverter(value, *args):
    if isinstance(value, float):
        return int(round(value, 0))
    return value