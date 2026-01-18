from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.vertex_attrib_integer_64bit import *
from OpenGL.raw.GL.NV.vertex_attrib_integer_64bit import _EXTENSION_NAME
def glInitVertexAttribInteger64BitNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)