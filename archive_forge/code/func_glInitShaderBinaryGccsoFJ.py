from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.FJ.shader_binary_GCCSO import *
from OpenGL.raw.GLES2.FJ.shader_binary_GCCSO import _EXTENSION_NAME
def glInitShaderBinaryGccsoFJ():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)