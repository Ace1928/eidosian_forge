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
def get_string_info(self, text, font_face):
    """Converts a string of text into a list of indices and advances used for shaping."""
    text_length = len(text.encode('utf-16-le')) // 2
    text_buffer = create_unicode_buffer(text, text_length)
    self._text_analysis.GenerateResults(self._analyzer, text_buffer, len(text_buffer))
    max_glyph_size = int(3 * text_length / 2 + 16)
    length = text_length
    clusters = (UINT16 * length)()
    text_props = (DWRITE_SHAPING_TEXT_PROPERTIES * length)()
    indices = (UINT16 * max_glyph_size)()
    glyph_props = (DWRITE_SHAPING_GLYPH_PROPERTIES * max_glyph_size)()
    actual_count = UINT32()
    self._analyzer.GetGlyphs(text_buffer, length, font_face, False, False, self._text_analysis._script, None, None, None, None, 0, max_glyph_size, clusters, text_props, indices, glyph_props, byref(actual_count))
    advances = (FLOAT * length)()
    offsets = (DWRITE_GLYPH_OFFSET * length)()
    self._analyzer.GetGlyphPlacements(text_buffer, clusters, text_props, text_length, indices, glyph_props, actual_count, font_face, self.font._font_metrics.designUnitsPerEm, False, False, self._text_analysis._script, self.font.locale, None, None, 0, advances, offsets)
    return (text_buffer, actual_count.value, indices, advances, offsets, clusters)