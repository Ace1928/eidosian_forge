from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLsizei, _cs.GLsizei, _cs.GLenum, _cs.GLsizei, ctypes.c_void_p)
def glCompressedTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, imageSize, data):
    pass