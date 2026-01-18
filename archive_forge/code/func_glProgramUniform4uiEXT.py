from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLint, _cs.GLuint, _cs.GLuint, _cs.GLuint, _cs.GLuint)
def glProgramUniform4uiEXT(program, location, v0, v1, v2, v3):
    pass