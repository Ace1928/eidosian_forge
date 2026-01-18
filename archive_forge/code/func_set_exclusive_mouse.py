from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
def set_exclusive_mouse(self, exclusive=True):
    if self._exclusive_mouse == exclusive and self._exclusive_mouse_focus == self._has_focus:
        return
    raw_mouse = RAWINPUTDEVICE(1, 2, 0, None)
    if not exclusive:
        raw_mouse.dwFlags = RIDEV_REMOVE
        raw_mouse.hwndTarget = None
    if not _user32.RegisterRawInputDevices(byref(raw_mouse), 1, sizeof(RAWINPUTDEVICE)):
        if exclusive:
            raise WindowException('Cannot enter mouse exclusive mode.')
    self._exclusive_mouse_buttons = 0
    if exclusive and self._has_focus:
        self._update_clipped_cursor()
    else:
        _user32.ClipCursor(None)
    self._exclusive_mouse = exclusive
    self._exclusive_mouse_focus = self._has_focus
    self.set_mouse_platform_visible(not exclusive)