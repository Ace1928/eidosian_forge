from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.SGIS.texture4D import *
from OpenGL.raw.GL.SGIS.texture4D import _EXTENSION_NAME
from OpenGL.GL import images as _i
def glInitTexture4DSGIS():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)