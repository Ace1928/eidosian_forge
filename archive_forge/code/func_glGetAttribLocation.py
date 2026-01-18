from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import *
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import _EXTENSION_NAME
from OpenGL import converters
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL._bytes import _NULL_8_BYTE
from OpenGL import contextdata 
@_lazy(glGetAttribLocation)
def glGetAttribLocation(baseOperation, program, name):
    """Check that name is a string with a null byte at the end of it"""
    if not name:
        raise ValueError('Non-null name required')
    name = as_8_bit(name)
    if name[-1] != _NULL_8_BYTE:
        name = name + _NULL_8_BYTE
    return baseOperation(program, name)