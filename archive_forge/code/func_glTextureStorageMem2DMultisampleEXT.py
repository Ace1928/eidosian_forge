from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLsizei, _cs.GLenum, _cs.GLsizei, _cs.GLsizei, _cs.GLboolean, _cs.GLuint, _cs.GLuint64)
def glTextureStorageMem2DMultisampleEXT(texture, samples, internalFormat, width, height, fixedSampleLocations, memory, offset):
    pass