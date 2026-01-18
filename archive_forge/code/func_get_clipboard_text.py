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
def get_clipboard_text(self) -> str:
    text = ''
    valid = _user32.OpenClipboard(self._view_hwnd)
    if not valid:
        print('Could not open clipboard')
        return ''
    cb_obj = _user32.GetClipboardData(CF_UNICODETEXT)
    if cb_obj:
        locked_data = _kernel32.GlobalLock(cb_obj)
        if locked_data:
            text = ctypes.wstring_at(locked_data)
            _kernel32.GlobalUnlock(cb_obj)
    _user32.CloseClipboard()
    return text