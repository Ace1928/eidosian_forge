from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, arrays.GLcharArray, _cs.GLsizei, arrays.GLintArray, arrays.GLcharArray)
def glGetNamedStringARB(namelen, name, bufSize, stringlen, string):
    pass