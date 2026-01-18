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
def get_glyphs(self, text):
    if not self._glyph_renderer:
        self._glyph_renderer = self.glyph_renderer_class(self)
        self._empty_glyph = self._glyph_renderer.render_using_layout(' ')
        self._zero_glyph = self._glyph_renderer.create_zero_glyph()
    text_buffer, actual_count, indices, advances, offsets, clusters = self._glyph_renderer.get_string_info(text, self.font_face)
    metrics = self._glyph_renderer.get_glyph_metrics(self.font_face, indices, actual_count)
    formatted_clusters = list(clusters)
    for i in range(actual_count):
        advances[i] *= self.font_scale_ratio
    for i in range(actual_count):
        offsets[i].advanceOffset *= self.font_scale_ratio
        offsets[i].ascenderOffset *= self.font_scale_ratio
    glyphs = []
    substitutions = {}
    for idx in clusters:
        ct = formatted_clusters.count(idx)
        if ct > 1:
            substitutions[idx] = ct - 1
    for i in range(actual_count):
        indice = indices[i]
        if indice == 0:
            glyph = self._render_layout_glyph(text_buffer, i, formatted_clusters)
            glyphs.append(glyph)
        else:
            advance_key = (indice, advances[i], offsets[i].advanceOffset, offsets[i].ascenderOffset)
            if indice in self.glyphs:
                if advance_key in self._advance_cache:
                    glyph = self._advance_cache[advance_key]
                else:
                    glyph = self.copy_glyph(self.glyphs[indice], advances[i], offsets[i])
                    self._advance_cache[advance_key] = glyph
            else:
                glyph = self._glyph_renderer.render_single_glyph(self.font_face, indice, advances[i], offsets[i], metrics[i])
                if glyph is None:
                    glyph = self._render_layout_glyph(text_buffer, i, formatted_clusters, check_color=False)
                    glyph.colored = True
                self.glyphs[indice] = glyph
                self._advance_cache[advance_key] = glyph
            glyphs.append(glyph)
        if i in substitutions:
            for _ in range(substitutions[i]):
                glyphs.append(self._zero_glyph)
    return glyphs