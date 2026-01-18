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
def set_clipboard_text(self, text: str):
    valid = _user32.OpenClipboard(self._view_hwnd)
    if not valid:
        return
    _user32.EmptyClipboard()
    size = (len(text) + 1) * sizeof(WCHAR)
    cb_data = _kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
    locked_data = _kernel32.GlobalLock(cb_data)
    memmove(locked_data, text, size)
    _kernel32.GlobalUnlock(cb_data)
    _user32.SetClipboardData(CF_UNICODETEXT, cb_data)
    _user32.CloseClipboard()