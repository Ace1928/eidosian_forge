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
def draw_cupcake(self):
    self.draw_label(self.no_issues_label, x=self.pad, y=self.pad, color=self.no_issues_label_color, width=self.width - 2 * self.pad, align=0.5, bold=True, font_size=self.title_font_size)
    self.draw_text(self.cupcake, x=self.pad, y=self.pad + self.font_size, width=self.width - 2 * self.pad, height=self.height - 2 * self.pad - self.font_size, color=self.cupcake_color)