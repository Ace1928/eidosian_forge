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
def dviFontName(self, dvifont):
    """
        Given a dvi font object, return a name suitable for Op.selectfont.
        This registers the font information in ``self.dviFontInfo`` if not yet
        registered.
        """
    dvi_info = self.dviFontInfo.get(dvifont.texname)
    if dvi_info is not None:
        return dvi_info.pdfname
    tex_font_map = dviread.PsfontsMap(dviread.find_tex_file('pdftex.map'))
    psfont = tex_font_map[dvifont.texname]
    if psfont.filename is None:
        raise ValueError('No usable font file found for {} (TeX: {}); the font may lack a Type-1 version'.format(psfont.psname, dvifont.texname))
    pdfname = next(self._internal_font_seq)
    _log.debug('Assigning font %s = %s (dvi)', pdfname, dvifont.texname)
    self.dviFontInfo[dvifont.texname] = types.SimpleNamespace(dvifont=dvifont, pdfname=pdfname, fontfile=psfont.filename, basefont=psfont.psname, encodingfile=psfont.encoding, effects=psfont.effects)
    return pdfname