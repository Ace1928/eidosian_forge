from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.MESA.shader_integer_functions import *
from OpenGL.raw.GLES2.MESA.shader_integer_functions import _EXTENSION_NAME
def glInitShaderIntegerFunctionsMESA():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)