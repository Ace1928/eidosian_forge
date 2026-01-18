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
@_lazy(glGetShaderSource)
def glGetShaderSource(baseOperation, obj):
    """Retrieve the program/shader's source code as a Python string

    returns string which is '' if no source code
    """
    length = int(glGetShaderiv(obj, GL_SHADER_SOURCE_LENGTH))
    if length > 0:
        source = ctypes.create_string_buffer(length)
        baseOperation(obj, length, None, source)
        return source.value.strip(_NULL_8_BYTE)
    return ''