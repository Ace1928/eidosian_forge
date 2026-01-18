from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_1_2 import *
from OpenGL.raw.GL.VERSION.GL_1_2 import _EXTENSION_NAME
from OpenGL.GL.ARB.imaging import *
from OpenGL.GL.VERSION.GL_1_2_images import *
def glInitGl12VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)