from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.IMG.texture_filter_cubic import *
from OpenGL.raw.GLES2.IMG.texture_filter_cubic import _EXTENSION_NAME
def glInitTextureFilterCubicIMG():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)