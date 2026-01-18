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
def get_system_mouse_cursor(self, name):
    if name == self.CURSOR_DEFAULT:
        return DefaultMouseCursor()
    names = {self.CURSOR_CROSSHAIR: IDC_CROSS, self.CURSOR_HAND: IDC_HAND, self.CURSOR_HELP: IDC_HELP, self.CURSOR_NO: IDC_NO, self.CURSOR_SIZE: IDC_SIZEALL, self.CURSOR_SIZE_UP: IDC_SIZENS, self.CURSOR_SIZE_UP_RIGHT: IDC_SIZENESW, self.CURSOR_SIZE_RIGHT: IDC_SIZEWE, self.CURSOR_SIZE_DOWN_RIGHT: IDC_SIZENWSE, self.CURSOR_SIZE_DOWN: IDC_SIZENS, self.CURSOR_SIZE_DOWN_LEFT: IDC_SIZENESW, self.CURSOR_SIZE_LEFT: IDC_SIZEWE, self.CURSOR_SIZE_UP_LEFT: IDC_SIZENWSE, self.CURSOR_SIZE_UP_DOWN: IDC_SIZENS, self.CURSOR_SIZE_LEFT_RIGHT: IDC_SIZEWE, self.CURSOR_TEXT: IDC_IBEAM, self.CURSOR_WAIT: IDC_WAIT, self.CURSOR_WAIT_ARROW: IDC_APPSTARTING}
    if name not in names:
        raise RuntimeError('Unknown cursor name "%s"' % name)
    cursor = _user32.LoadCursorW(None, MAKEINTRESOURCE(names[name]))
    return Win32MouseCursor(cursor)