from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.NV.vertex_array_range import *
from OpenGL.raw.WGL.NV.vertex_array_range import _EXTENSION_NAME
def glInitVertexArrayRangeNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)