from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.parallel_shader_compile import *
from OpenGL.raw.GL.ARB.parallel_shader_compile import _EXTENSION_NAME
def glInitParallelShaderCompileARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)