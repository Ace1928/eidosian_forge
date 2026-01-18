from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.compute_variable_group_size import *
from OpenGL.raw.GL.ARB.compute_variable_group_size import _EXTENSION_NAME
def glInitComputeVariableGroupSizeARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)