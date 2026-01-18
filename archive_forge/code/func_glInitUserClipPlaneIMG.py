from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES1 import _types, _glgets
from OpenGL.raw.GLES1.IMG.user_clip_plane import *
from OpenGL.raw.GLES1.IMG.user_clip_plane import _EXTENSION_NAME
def glInitUserClipPlaneIMG():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)