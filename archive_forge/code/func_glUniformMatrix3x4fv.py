from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES3 import _types as _cs
from OpenGL.raw.GLES3._types import *
from OpenGL.raw.GLES3 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, _cs.GLsizei, _cs.GLboolean, arrays.GLfloatArray)
def glUniformMatrix3x4fv(location, count, transpose, value):
    pass