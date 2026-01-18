from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, arrays.GLuintArray, arrays.GLintArray, arrays.GLsizeiArray, _cs.GLsizei, _cs.GLint)
def glMultiModeDrawArraysIBM(mode, first, count, primcount, modestride):
    pass