from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
from OpenGL.raw.GL.VERSION.GL_1_0 import *
import ctypes
@_f
@_p.types(None, _cs.GLenum, _cs.GLint, _cs.GLenum, _cs.GLint, _cs.GLint, _cs.GLsizei, _cs.GLint)
def glCopyTexImage1D(target, level, internalformat, x, y, width, border):
    pass