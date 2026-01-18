from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.win32 import _user32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32.context_managers import device_context
def enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
    r = lprcMonitor.contents
    width = r.right - r.left
    height = r.bottom - r.top
    screens.append(Win32Screen(self, hMonitor, r.left, r.top, width, height))
    return True