import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
@current_position.setter
def current_position(self, value):
    _check(self._native_buffer.SetCurrentPosition(value))