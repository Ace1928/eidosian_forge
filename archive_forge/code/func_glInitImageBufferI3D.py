from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.I3D.image_buffer import *
from OpenGL.raw.WGL.I3D.image_buffer import _EXTENSION_NAME
def glInitImageBufferI3D():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)