from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.NV.conservative_raster import *
from OpenGL.raw.GLES2.NV.conservative_raster import _EXTENSION_NAME
def glInitConservativeRasterNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)