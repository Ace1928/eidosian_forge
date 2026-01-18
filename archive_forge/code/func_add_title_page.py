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
def add_title_page(self, files, *, show_tolerance=True, tolerance=None, kinkiness=None):
    pad = self.pad
    width = self.width - 3 * self.pad
    height = self.height - 2 * self.pad
    x = y = pad
    self.draw_label('Problem report for:', x=x, y=y, bold=True, width=width, font_size=self.title_font_size)
    y += self.title_font_size
    import hashlib
    for file in files:
        base_file = os.path.basename(file)
        y += self.font_size + self.pad
        self.draw_label(base_file, x=x, y=y, bold=True, width=width)
        y += self.font_size + self.pad
        try:
            h = hashlib.sha1(open(file, 'rb').read()).hexdigest()
            self.draw_label('sha1: %s' % h, x=x + pad, y=y, width=width)
            y += self.font_size
        except IsADirectoryError:
            pass
        if file.endswith('.ttf'):
            ttFont = TTFont(file)
            name = ttFont['name'] if 'name' in ttFont else None
            if name:
                for what, nameIDs in (('Family name', (21, 16, 1)), ('Version', (5,))):
                    n = name.getFirstDebugName(nameIDs)
                    if n is None:
                        continue
                    self.draw_label('%s: %s' % (what, n), x=x + pad, y=y, width=width)
                    y += self.font_size + self.pad
        elif file.endswith(('.glyphs', '.glyphspackage')):
            from glyphsLib import GSFont
            f = GSFont(file)
            for what, field in (('Family name', 'familyName'), ('VersionMajor', 'versionMajor'), ('VersionMinor', '_versionMinor')):
                self.draw_label('%s: %s' % (what, getattr(f, field)), x=x + pad, y=y, width=width)
                y += self.font_size + self.pad
    self.draw_legend(show_tolerance=show_tolerance, tolerance=tolerance, kinkiness=kinkiness)
    self.show_page()