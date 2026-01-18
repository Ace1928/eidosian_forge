from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.APPLE.texture_format_BGRA8888 import *
from OpenGL.raw.GLES2.APPLE.texture_format_BGRA8888 import _EXTENSION_NAME
def glInitTextureFormatBgra8888APPLE():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)