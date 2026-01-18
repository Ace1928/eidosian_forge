from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.fragment_program_option import *
from OpenGL.raw.GL.NV.fragment_program_option import _EXTENSION_NAME
def glInitFragmentProgramOptionNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)