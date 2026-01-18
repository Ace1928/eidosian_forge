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
def is_fallback_str_colored(self, font_face, text):
    indice = UINT16()
    code_points = (UINT32 * len(text))(*[ord(c) for c in text])
    font_face.GetGlyphIndices(code_points, len(text), byref(indice))
    new_indice = (UINT16 * 1)(indice)
    new_advance = (FLOAT * 1)(100)
    offset = (DWRITE_GLYPH_OFFSET * 1)()
    run = self._glyph_renderer._get_single_glyph_run(font_face, self._real_size, new_indice, new_advance, offset, False, False)
    return self._glyph_renderer.is_color_run(run)