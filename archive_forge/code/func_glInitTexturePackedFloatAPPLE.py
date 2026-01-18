from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.APPLE.texture_packed_float import *
from OpenGL.raw.GLES2.APPLE.texture_packed_float import _EXTENSION_NAME
def glInitTexturePackedFloatAPPLE():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)