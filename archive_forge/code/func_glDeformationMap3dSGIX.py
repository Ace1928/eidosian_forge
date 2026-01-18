from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLenum, _cs.GLdouble, _cs.GLdouble, _cs.GLint, _cs.GLint, _cs.GLdouble, _cs.GLdouble, _cs.GLint, _cs.GLint, _cs.GLdouble, _cs.GLdouble, _cs.GLint, _cs.GLint, arrays.GLdoubleArray)
def glDeformationMap3dSGIX(target, u1, u2, ustride, uorder, v1, v2, vstride, vorder, w1, w2, wstride, worder, points):
    pass