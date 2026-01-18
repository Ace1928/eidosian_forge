from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_1_5 import *
from OpenGL.raw.GL.VERSION.GL_1_5 import _EXTENSION_NAME
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.arrays import ArrayDatatype
from OpenGL._bytes import integer_types
def glInitGl15VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)