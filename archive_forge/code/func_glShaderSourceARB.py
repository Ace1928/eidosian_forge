from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLhandleARB, _cs.GLsizei, ctypes.POINTER(ctypes.POINTER(_cs.GLchar)), arrays.GLintArray)
def glShaderSourceARB(shaderObj, count, string, length):
    pass