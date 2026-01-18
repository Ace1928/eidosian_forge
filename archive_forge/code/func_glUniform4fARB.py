from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat)
def glUniform4fARB(location, v0, v1, v2, v3):
    pass