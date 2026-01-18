import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
@staticmethod
def get_localized_index(strings: IDWriteLocalizedStrings, locale: str):
    idx = UINT32()
    exists = BOOL()
    if locale:
        strings.FindLocaleName(locale, byref(idx), byref(exists))
        if not exists.value:
            strings.FindLocaleName('en-us', byref(idx), byref(exists))
            if not exists:
                return 0
        return idx.value
    return 0