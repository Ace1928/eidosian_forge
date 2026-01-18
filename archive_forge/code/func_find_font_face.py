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
@classmethod
def find_font_face(cls, font_name, bold, italic, stretch) -> Tuple[Optional[IDWriteFont], Optional[IDWriteFontCollection]]:
    """This will search font collections for legacy RBIZ names. However, matching to bold, italic, stretch is
        problematic in that there are many values. We parse the font name looking for matches to the name database,
        and attempt to pick the closest match.
        This will search all fonts on the system and custom loaded, and all of their font faces. Returns a collection
        and IDWriteFont if successful.
        """
    p_bold, p_italic, p_stretch = cls.parse_name(font_name, bold, italic, stretch)
    _debug_print(f"directwrite: '{font_name}' not found. Attempting legacy name lookup in all collections.")
    collection_idx = cls.find_legacy_font(cls._custom_collection, font_name, p_bold, p_italic, p_stretch)
    if collection_idx is not None:
        return (collection_idx, cls._custom_collection)
    sys_collection = IDWriteFontCollection()
    cls._write_factory.GetSystemFontCollection(byref(sys_collection), 1)
    collection_idx = cls.find_legacy_font(sys_collection, font_name, p_bold, p_italic, p_stretch)
    if collection_idx is not None:
        return (collection_idx, sys_collection)
    return (None, None)