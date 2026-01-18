from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.vertex_program import *
from OpenGL.raw.GL.ARB.vertex_program import _EXTENSION_NAME
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import converters, error, contextdata
from OpenGL.arrays.arraydatatype import ArrayDatatype
def glInitVertexProgramARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)