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
@staticmethod
def parse_filenames(decoded_string):
    """All of the filenames from file drops come as one big string with
            some special characters (%20), this will parse them out.
        """
    import sys
    different_files = decoded_string.splitlines()
    parsed = []
    for filename in different_files:
        if filename:
            filename = urllib.parse.urlsplit(filename).path
            encoding = sys.getfilesystemencoding()
            parsed.append(urllib.parse.unquote(filename, encoding))
    return parsed