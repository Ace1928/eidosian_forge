from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.MESA.pack_invert import *
from OpenGL.raw.GL.MESA.pack_invert import _EXTENSION_NAME
def glInitPackInvertMESA():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)