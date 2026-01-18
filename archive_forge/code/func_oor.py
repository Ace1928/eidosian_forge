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
def oor(*args):
    if len(args) > 1:
        oor = self.originalObject(args[1])
        return function(args[0], oor)
    else:
        return function(args[0])