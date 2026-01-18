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
def draw_circle(self, cr, *, x=0, y=0, color=(0, 0, 0), diameter=10, stroke_width=1):
    cr.save()
    cr.set_line_width(stroke_width)
    cr.set_line_cap(cairo.LINE_CAP_SQUARE)
    cr.arc(x, y, diameter / 2, 0, 2 * math.pi)
    if len(color) == 3:
        color = color + (1,)
    cr.set_source_rgba(*color)
    cr.stroke()
    cr.restore()