from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLuint, _cs.GLsizei, arrays.GLsizeiArray, arrays.GLsizeiArray, arrays.GLuintArray, arrays.GLcharArray)
def glGetTransformFeedbackVaryingEXT(program, index, bufSize, length, size, type, name):
    pass