from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.EXT.make_current_read import *
from OpenGL.raw.WGL.EXT.make_current_read import _EXTENSION_NAME
def glInitMakeCurrentReadEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)