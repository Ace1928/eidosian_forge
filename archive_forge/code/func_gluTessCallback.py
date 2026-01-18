from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def gluTessCallback(tess, which, function):
    """Set a given gluTessellator callback for the given tessellator"""
    return tess.addCallback(which, function)