from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.texture_sRGB_R8 import *
from OpenGL.raw.GLES2.EXT.texture_sRGB_R8 import _EXTENSION_NAME
def glInitTextureSrgbR8EXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)