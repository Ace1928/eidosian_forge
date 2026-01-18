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
def embedTTFType3(font, characters, descriptor):
    """The Type 3-specific part of embedding a Truetype font"""
    widthsObject = self.reserveObject('font widths')
    fontdescObject = self.reserveObject('font descriptor')
    fontdictObject = self.reserveObject('font dictionary')
    charprocsObject = self.reserveObject('character procs')
    differencesArray = []
    firstchar, lastchar = (0, 255)
    bbox = [cvt(x, nearest=False) for x in font.bbox]
    fontdict = {'Type': Name('Font'), 'BaseFont': ps_name, 'FirstChar': firstchar, 'LastChar': lastchar, 'FontDescriptor': fontdescObject, 'Subtype': Name('Type3'), 'Name': descriptor['FontName'], 'FontBBox': bbox, 'FontMatrix': [0.001, 0, 0, 0.001, 0, 0], 'CharProcs': charprocsObject, 'Encoding': {'Type': Name('Encoding'), 'Differences': differencesArray}, 'Widths': widthsObject}
    from encodings import cp1252

    def get_char_width(charcode):
        s = ord(cp1252.decoding_table[charcode])
        width = font.load_char(s, flags=LOAD_NO_SCALE | LOAD_NO_HINTING).horiAdvance
        return cvt(width)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        widths = [get_char_width(charcode) for charcode in range(firstchar, lastchar + 1)]
    descriptor['MaxWidth'] = max(widths)
    glyph_ids = []
    differences = []
    multi_byte_chars = set()
    for c in characters:
        ccode = c
        gind = font.get_char_index(ccode)
        glyph_ids.append(gind)
        glyph_name = font.get_glyph_name(gind)
        if ccode <= 255:
            differences.append((ccode, glyph_name))
        else:
            multi_byte_chars.add(glyph_name)
    differences.sort()
    last_c = -2
    for c, name in differences:
        if c != last_c + 1:
            differencesArray.append(c)
        differencesArray.append(Name(name))
        last_c = c
    rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
    charprocs = {}
    for charname in sorted(rawcharprocs):
        stream = rawcharprocs[charname]
        charprocDict = {}
        if charname in multi_byte_chars:
            charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
            stream = stream[stream.find(b'd1') + 2:]
        charprocObject = self.reserveObject('charProc')
        self.outputStream(charprocObject, stream, extra=charprocDict)
        if charname in multi_byte_chars:
            name = self._get_xobject_glyph_name(filename, charname)
            self.multi_byte_charprocs[name] = charprocObject
        else:
            charprocs[charname] = charprocObject
    self.writeObject(fontdictObject, fontdict)
    self.writeObject(fontdescObject, descriptor)
    self.writeObject(widthsObject, widths)
    self.writeObject(charprocsObject, charprocs)
    return fontdictObject