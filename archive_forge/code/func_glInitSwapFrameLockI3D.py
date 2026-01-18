from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.I3D.swap_frame_lock import *
from OpenGL.raw.WGL.I3D.swap_frame_lock import _EXTENSION_NAME
def glInitSwapFrameLockI3D():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)