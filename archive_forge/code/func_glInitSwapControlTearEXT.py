from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.EXT.swap_control_tear import *
from OpenGL.raw.WGL.EXT.swap_control_tear import _EXTENSION_NAME
def glInitSwapControlTearEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)