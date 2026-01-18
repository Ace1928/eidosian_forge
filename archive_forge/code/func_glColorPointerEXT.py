from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, _cs.GLenum, _cs.GLsizei, _cs.GLsizei, ctypes.c_void_p)
def glColorPointerEXT(size, type, stride, count, pointer):
    pass