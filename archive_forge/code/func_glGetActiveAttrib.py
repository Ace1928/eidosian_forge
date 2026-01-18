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
@_lazy(glGetActiveAttrib)
def glGetActiveAttrib(baseOperation, program, index, bufSize=None, *args):
    """Retrieves information about the attribute variable.

    program -- specifies the program to be queried
    index -- index of the attribute to be queried 
    
    Following parameters are optional:
    
    bufSize -- determines the size of the buffer (limits number of bytes written),
               if not provided, will be GL_ACTIVE_ATTRIBUTE_MAX_LENGTH
    length -- pointer-to-GLsizei that will hold the resulting length of the name
    size -- pointer-to-GLint that will hold the size of the attribute
    type -- pointer-to-GLenum that will hold the type constant of the attribute
    name -- pointer-to-GLchar that will hold the (null-terminated) name string
    
    returns (bytes) name, (int)size, (enum)type
    """
    if bufSize is None:
        bufSize = int(glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH))
    if bufSize <= 0:
        raise RuntimeError('Active attribute length reported', bufsize)
    name, size, type = baseOperation(program, index, bufSize, *args)[1:]
    if hasattr(name, 'tostring'):
        name = name.tostring().rstrip(b'\x00')
    elif hasattr(name, 'value'):
        name = name.value
    return (name, size, type)