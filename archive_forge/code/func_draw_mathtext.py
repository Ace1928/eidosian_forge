import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
@_log_if_debug_on
def draw_mathtext(self, gc, x, y, s, prop, angle):
    """Draw the math text using matplotlib.mathtext."""
    width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, 72, prop)
    self.set_color(*gc.get_rgb())
    self._pswriter.write(f'gsave\n{x:g} {y:g} translate\n{angle:g} rotate\n')
    lastfont = None
    for font, fontsize, num, ox, oy in glyphs:
        self._character_tracker.track_glyph(font, num)
        if (font.postscript_name, fontsize) != lastfont:
            lastfont = (font.postscript_name, fontsize)
            self._pswriter.write(f'/{font.postscript_name} {fontsize} selectfont\n')
        glyph_name = font.get_name_char(chr(num)) if isinstance(font, AFM) else font.get_glyph_name(font.get_char_index(num))
        self._pswriter.write(f'{ox:g} {oy:g} moveto\n/{glyph_name} glyphshow\n')
    for ox, oy, w, h in rects:
        self._pswriter.write(f'{ox} {oy} {w} {h} rectfill\n')
    self._pswriter.write('grestore\n')