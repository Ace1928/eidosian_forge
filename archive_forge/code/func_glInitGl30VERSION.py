from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_3_0 import *
from OpenGL.raw.GL.VERSION.GL_3_0 import _EXTENSION_NAME
from ctypes import c_char_p
def glInitGl30VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)