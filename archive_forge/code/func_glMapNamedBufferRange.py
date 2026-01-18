from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(ctypes.c_void_p, _cs.GLuint, _cs.GLintptr, _cs.GLsizeiptr, _cs.GLbitfield)
def glMapNamedBufferRange(buffer, offset, length, access):
    pass