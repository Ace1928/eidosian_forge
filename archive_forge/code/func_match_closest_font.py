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
def match_closest_font(font_list: List[Tuple[int, int, int, IDWriteFont]], bold: int, italic: int, stretch: int) -> Optional[IDWriteFont]:
    """Match the closest font to the parameters specified. If a full match is not found, a secondary match will be
        found based on similar features. This can probably be improved, but it is possible you could get a different
        font style than expected."""
    closest = []
    for match in font_list:
        f_weight, f_style, f_stretch, writefont = match
        if f_weight == bold and f_style == italic and (f_stretch == stretch):
            _debug_print(f'directwrite: full match found. (bold: {f_weight}, italic: {f_style}, stretch: {f_stretch})')
            return writefont
        prop_match = 0
        similar_match = 0
        if f_weight == bold:
            prop_match += 1
        elif bold != DWRITE_FONT_WEIGHT_NORMAL and f_weight != DWRITE_FONT_WEIGHT_NORMAL:
            similar_match += 1
        if f_style == italic:
            prop_match += 1
        elif italic != DWRITE_FONT_STYLE_NORMAL and f_style != DWRITE_FONT_STYLE_NORMAL:
            similar_match += 1
        if stretch == f_stretch:
            prop_match += 1
        elif stretch != DWRITE_FONT_STRETCH_NORMAL and f_stretch != DWRITE_FONT_STRETCH_NORMAL:
            similar_match += 1
        closest.append((prop_match, similar_match, *match))
    closest.sort(key=lambda fts: (fts[0], fts[1]), reverse=True)
    if closest:
        closest_match = closest[0]
        _debug_print(f'directwrite: falling back to partial match. (bold: {closest_match[2]}, italic: {closest_match[3]}, stretch: {closest_match[4]})')
        return closest_match[5]
    return None