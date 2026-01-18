from OpenGL import platform as _p, arrays
from OpenGL.raw.GLX import _types as _cs
from OpenGL.raw.GLX._types import *
from OpenGL.raw.GLX import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLXContext, ctypes.POINTER(_cs.Display), _cs.GLXContextID)
def glXImportContextEXT(dpy, contextID):
    pass