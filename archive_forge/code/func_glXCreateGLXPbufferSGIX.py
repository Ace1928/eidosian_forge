from OpenGL import platform as _p, arrays
from OpenGL.raw.GLX import _types as _cs
from OpenGL.raw.GLX._types import *
from OpenGL.raw.GLX import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLXPbufferSGIX, ctypes.POINTER(_cs.Display), _cs.GLXFBConfigSGIX, _cs.c_uint, _cs.c_uint, ctypes.POINTER(_cs.c_int))
def glXCreateGLXPbufferSGIX(dpy, config, width, height, attrib_list):
    pass