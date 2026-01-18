from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.blend_func_separate import *
from OpenGL.raw.GL.EXT.blend_func_separate import _EXTENSION_NAME
def glInitBlendFuncSeparateEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)