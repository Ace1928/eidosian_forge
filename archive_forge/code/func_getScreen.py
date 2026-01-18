from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
def getScreen(self, display):
    from OpenGL.platform import ctypesloader
    from OpenGL.raw.GLX import _types
    import ctypes, os
    X11 = ctypesloader.loadLibrary(ctypes.cdll, 'X11')
    XDefaultScreen = X11.XDefaultScreen
    XDefaultScreen.argtypes = [ctypes.POINTER(_types.Display)]
    return XDefaultScreen(display)