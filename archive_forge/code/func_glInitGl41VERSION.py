from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_4_1 import *
from OpenGL.raw.GL.VERSION.GL_4_1 import _EXTENSION_NAME
from OpenGL.GL.ARB.ES2_compatibility import *
from OpenGL.GL.ARB.get_program_binary import *
from OpenGL.GL.ARB.separate_shader_objects import *
from OpenGL.GL.ARB.shader_precision import *
from OpenGL.GL.ARB.vertex_attrib_64bit import *
from OpenGL.GL.ARB.viewport_array import *
def glInitGl41VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)