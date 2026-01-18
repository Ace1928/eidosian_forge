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
def add_problems(self, problems, *, show_tolerance=True, show_page_number=True):
    for glyph, glyph_problems in problems.items():
        last_masters = None
        current_glyph_problems = []
        for p in glyph_problems:
            masters = p['master_idx'] if 'master_idx' in p else (p['master_1_idx'], p['master_2_idx'])
            if masters == last_masters:
                current_glyph_problems.append(p)
                continue
            if current_glyph_problems:
                self.add_problem(glyph, current_glyph_problems, show_tolerance=show_tolerance, show_page_number=show_page_number)
                self.show_page()
                current_glyph_problems = []
            last_masters = masters
            current_glyph_problems.append(p)
        if current_glyph_problems:
            self.add_problem(glyph, current_glyph_problems, show_tolerance=show_tolerance, show_page_number=show_page_number)
            self.show_page()