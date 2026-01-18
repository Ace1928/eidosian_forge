from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES1 import _types as _cs
from OpenGL.raw.GLES1._types import *
from OpenGL.raw.GLES1 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, _cs.GLfixed, _cs.GLfixed, _cs.GLfixed, _cs.GLfixed)
def glMapGrid2xOES(n, u1, u2, v1, v2):
    pass