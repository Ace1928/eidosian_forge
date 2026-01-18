from OpenGL.raw import GLU as _simple
from OpenGL.platform import createBaseFunction, PLATFORM
import ctypes
def gluQuadricCallback(quadric, which=_simple.GLU_ERROR, function=None):
    """Set the GLU error callback function"""
    return quadric.addCallback(which, function)