from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.texture_cube_map_array import *
from OpenGL.raw.GL.ARB.texture_cube_map_array import _EXTENSION_NAME
def glInitTextureCubeMapArrayARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)