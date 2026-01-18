from OpenGL import platform as _p, arrays
from OpenGL.raw.EGL import _types as _cs
from OpenGL.raw.EGL._types import *
from OpenGL.raw.EGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.EGLBoolean, _cs.EGLDisplay, _cs.EGLSyncKHR, _cs.EGLint, arrays.GLintArray)
def eglGetSyncAttribKHR(dpy, sync, attribute, value):
    pass