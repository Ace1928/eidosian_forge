from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.VERSION.EGL_1_5 import *
from OpenGL.raw.EGL.VERSION.EGL_1_5 import _EXTENSION_NAME
from ..EXT.platform_base import eglGetPlatformDisplayEXT
from OpenGL.extensions import alternate as _alternate
def glInitEgl15VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)