from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.texture_format_sRGB_override import *
from OpenGL.raw.GLES2.EXT.texture_format_sRGB_override import _EXTENSION_NAME
def glInitTextureFormatSrgbOverrideEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)