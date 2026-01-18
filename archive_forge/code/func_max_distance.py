import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
@max_distance.setter
def max_distance(self, value):
    if self.is3d:
        _check(self._native_buffer3d.SetMaxDistance(value, lib.DS3D_IMMEDIATE))