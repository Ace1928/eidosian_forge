from OpenGL import platform as _p, arrays
from OpenGL.raw.EGL import _types as _cs
from OpenGL.raw.EGL._types import *
from OpenGL.raw.EGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(ctypes.c_char_p, _cs.EGLDisplay, _cs.EGLOutputPortEXT, _cs.EGLint)
def eglQueryOutputPortStringEXT(dpy, port, name):
    pass