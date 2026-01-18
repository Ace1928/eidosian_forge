from OpenGL.raw import GLU as _simple
from OpenGL import platform, converters, wrapper
from OpenGL.GLU import glustruct
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import arrays, error
import ctypes
import weakref
from OpenGL.platform import PLATFORM
import OpenGL
from OpenGL import _configflags
@_lazy(_simple.gluNewNurbsRenderer)
def gluNewNurbsRenderer(baseFunction):
    """Return a new nurbs renderer for the system (dereferences pointer)"""
    newSet = baseFunction()
    new = newSet[0]
    return new