from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.stencil_clear_tag import *
from OpenGL.raw.GL.EXT.stencil_clear_tag import _EXTENSION_NAME
def glInitStencilClearTagEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)