from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLuint, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat)
def glDrawTextureNV(texture, sampler, x0, y0, x1, y1, z, s0, t0, s1, t1):
    pass