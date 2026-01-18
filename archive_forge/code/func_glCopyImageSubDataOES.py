from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLenum, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLuint, _cs.GLenum, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLsizei, _cs.GLsizei, _cs.GLsizei)
def glCopyImageSubDataOES(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth):
    pass