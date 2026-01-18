import locale
import unicodedata
import urllib.parse
from ctypes import *
from functools import lru_cache
from typing import Optional
import pyglet
from pyglet.window import WindowException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor
from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import key
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.canvas.xlib import XlibCanvas
from pyglet.libs.x11 import xlib
from pyglet.libs.x11 import cursorfont
from pyglet.util import asbytes
def dispatch_platform_event(self, e):
    if self._applied_mouse_exclusive is None:
        self._update_exclusivity()
    event_handler = self._event_handlers.get(e.type)
    if event_handler:
        event_handler(e)