from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLenum, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint)
def glEvalMesh2(mode, i1, i2, j1, j2):
    pass