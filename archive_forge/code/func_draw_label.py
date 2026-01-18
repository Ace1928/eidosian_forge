from .interpolatableHelpers import *
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.recordingPen import (
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.pens.pointPen import (
from fontTools.varLib.interpolatableHelpers import (
from itertools import cycle
from functools import wraps
from io import BytesIO
import cairo
import math
import os
import logging
def draw_label(self, label, *, x=0, y=0, color=(0, 0, 0), align=0, bold=False, width=None, height=None, font_size=None):
    if width is None:
        width = self.width
    if height is None:
        height = self.height
    if font_size is None:
        font_size = self.font_size
    cr = cairo.Context(self.surface)
    cr.select_font_face('@cairo:', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(font_size)
    font_extents = cr.font_extents()
    font_size = font_size * font_size / font_extents[2]
    cr.set_font_size(font_size)
    font_extents = cr.font_extents()
    cr.set_source_rgb(*color)
    extents = cr.text_extents(label)
    if extents.width > width:
        font_size *= width / extents.width
        cr.set_font_size(font_size)
        font_extents = cr.font_extents()
        extents = cr.text_extents(label)
    label_x = x + (width - extents.width) * align
    label_y = y + font_extents[0]
    cr.move_to(label_x, label_y)
    cr.show_text(label)