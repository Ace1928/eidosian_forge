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
def set_mouse_platform_visible(self, platform_visible=None):
    if platform_visible is None:
        platform_visible = self._mouse_visible and (not self._exclusive_mouse) and (not self._mouse_cursor.gl_drawable or self._mouse_cursor.hw_drawable) or (not self._mouse_in_window or not self._has_focus)
    if platform_visible and self._mouse_cursor.hw_drawable:
        if isinstance(self._mouse_cursor, Win32MouseCursor):
            cursor = self._mouse_cursor.cursor
        elif isinstance(self._mouse_cursor, DefaultMouseCursor):
            cursor = _user32.LoadCursorW(None, MAKEINTRESOURCE(IDC_ARROW))
        else:
            cursor = self._create_cursor_from_image(self._mouse_cursor)
        _user32.SetClassLongPtrW(self._view_hwnd, GCL_HCURSOR, cursor)
        _user32.SetCursor(cursor)
    if platform_visible == self._mouse_platform_visible:
        return
    self._set_cursor_visibility(platform_visible)
    self._mouse_platform_visible = platform_visible