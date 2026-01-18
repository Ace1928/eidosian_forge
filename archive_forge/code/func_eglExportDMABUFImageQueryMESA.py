from OpenGL import platform as _p, arrays
from OpenGL.raw.EGL import _types as _cs
from OpenGL.raw.EGL._types import *
from OpenGL.raw.EGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.EGLBoolean, _cs.EGLDisplay, _cs.EGLImageKHR, ctypes.POINTER(_cs.c_int), ctypes.POINTER(_cs.c_int), arrays.GLuint64Array)
def eglExportDMABUFImageQueryMESA(dpy, image, fourcc, num_planes, modifiers):
    pass