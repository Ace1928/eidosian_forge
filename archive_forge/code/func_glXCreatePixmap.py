from OpenGL import platform as _p, arrays
from OpenGL.raw.GLX import _types as _cs
from OpenGL.raw.GLX._types import *
from OpenGL.raw.GLX import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLXPixmap, ctypes.POINTER(_cs.Display), _cs.GLXFBConfig, _cs.Pixmap, ctypes.POINTER(_cs.c_int))
def glXCreatePixmap(dpy, config, pixmap, attrib_list):
    pass