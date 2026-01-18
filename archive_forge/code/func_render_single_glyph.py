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
def render_single_glyph(self, font_face, indice, advance, offset, metrics):
    """Renders a single glyph using D2D DrawGlyphRun"""
    glyph_width, glyph_height, glyph_lsb, glyph_advance, glyph_bsb = metrics
    new_indice = (UINT16 * 1)(indice)
    new_advance = (FLOAT * 1)(advance)
    run = self._get_single_glyph_run(font_face, self.font._real_size, new_indice, new_advance, pointer(offset), False, 0)
    if self.draw_options & D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT and self.is_color_run(run):
        return None
    if glyph_advance:
        render_width = int(math.ceil(glyph_advance * self.font.font_scale_ratio))
    else:
        render_width = int(math.ceil(glyph_width * self.font.font_scale_ratio))
    render_offset_x = 0
    if glyph_lsb < 0:
        render_offset_x = glyph_lsb * self.font.font_scale_ratio
    if self.font.italic:
        render_width += render_width // 2
    self._create_bitmap(render_width + 1, int(math.ceil(self.font.max_glyph_height)))
    baseline_offset = D2D_POINT_2F(-render_offset_x - offset.advanceOffset, self.font.ascent + offset.ascenderOffset)
    self._render_target.BeginDraw()
    self._render_target.Clear(transparent)
    self._render_target.DrawGlyphRun(baseline_offset, run, self._brush, self.measuring_mode)
    self._render_target.EndDraw(None, None)
    image = wic_decoder.get_image(self._bitmap)
    glyph = self.font.create_glyph(image)
    glyph.set_bearings(-self.font.descent, render_offset_x, advance, offset.advanceOffset, offset.ascenderOffset)
    return glyph