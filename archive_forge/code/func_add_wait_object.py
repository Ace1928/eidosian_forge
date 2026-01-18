import ctypes
from .base import PlatformEventLoop
from pyglet.libs.win32 import _kernel32, _user32, types, constants
from pyglet.libs.win32.types import *
def add_wait_object(self, obj, func):
    self._wait_objects.append((obj, func))
    self._recreate_wait_objects_array()