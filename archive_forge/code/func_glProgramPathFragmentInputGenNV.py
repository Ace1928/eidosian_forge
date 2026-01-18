from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLint, _cs.GLenum, _cs.GLint, arrays.GLfloatArray)
def glProgramPathFragmentInputGenNV(program, location, genMode, components, coeffs):
    pass