from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.shader_objects import *
from OpenGL.raw.GL.ARB.shader_objects import _EXTENSION_NAME
import OpenGL
from OpenGL._bytes import bytes, _NULL_8_BYTE, as_8_bit
from OpenGL.raw.GL import _errors
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import converters, error
def glInitShaderObjectsARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)