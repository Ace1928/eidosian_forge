from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.AMD.gpu_shader_int64 import *
from OpenGL.raw.GL.AMD.gpu_shader_int64 import _EXTENSION_NAME
def glInitGpuShaderInt64AMD():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)