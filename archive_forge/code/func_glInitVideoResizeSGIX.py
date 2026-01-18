from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLX import _types, _glgets
from OpenGL.raw.GLX.SGIX.video_resize import *
from OpenGL.raw.GLX.SGIX.video_resize import _EXTENSION_NAME
def glInitVideoResizeSGIX():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)