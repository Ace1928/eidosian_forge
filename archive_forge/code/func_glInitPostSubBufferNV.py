from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.NV.post_sub_buffer import *
from OpenGL.raw.EGL.NV.post_sub_buffer import _EXTENSION_NAME
def glInitPostSubBufferNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)