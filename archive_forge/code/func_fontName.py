import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def fontName(self, fontprop):
    """
        Select a font based on fontprop and return a name suitable for
        Op.selectfont. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """
    if isinstance(fontprop, str):
        filenames = [fontprop]
    elif mpl.rcParams['pdf.use14corefonts']:
        filenames = _fontManager._find_fonts_by_props(fontprop, fontext='afm', directory=RendererPdf._afm_font_dir)
    else:
        filenames = _fontManager._find_fonts_by_props(fontprop)
    first_Fx = None
    for fname in filenames:
        Fx = self.fontNames.get(fname)
        if not first_Fx:
            first_Fx = Fx
        if Fx is None:
            Fx = next(self._internal_font_seq)
            self.fontNames[fname] = Fx
            _log.debug('Assigning font %s = %r', Fx, fname)
            if not first_Fx:
                first_Fx = Fx
    return first_Fx