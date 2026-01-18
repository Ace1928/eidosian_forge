from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.ARB.create_context_robustness import *
from OpenGL.raw.WGL.ARB.create_context_robustness import _EXTENSION_NAME
def glInitCreateContextRobustnessARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)