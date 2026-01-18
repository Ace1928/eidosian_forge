from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.OML.subsample import *
from OpenGL.raw.GL.OML.subsample import _EXTENSION_NAME
from OpenGL import images as _i
def glInitSubsampleOML():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)