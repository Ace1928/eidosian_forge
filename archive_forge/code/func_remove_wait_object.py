import ctypes
from .base import PlatformEventLoop
from pyglet.libs.win32 import _kernel32, _user32, types, constants
from pyglet.libs.win32.types import *
def remove_wait_object(self, obj):
    for i, (_object, _) in enumerate(self._wait_objects):
        if obj == _object:
            del self._wait_objects[i]
            break
    self._recreate_wait_objects_array()