import ctypes
from ctypes import *
import pyglet
from pyglet.gl.lib import missing_function, decorate_function
from pyglet.util import asbytes
def makeWGLFunction(func):

    class WGLFunction:
        __slots__ = class_slots
        __call__ = func
    return WGLFunction