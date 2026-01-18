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
def createType1Descriptor(self, t1font, fontfile):
    fontdescObject = self.reserveObject('font descriptor')
    fontfileObject = self.reserveObject('font file')
    italic_angle = t1font.prop['ItalicAngle']
    fixed_pitch = t1font.prop['isFixedPitch']
    flags = 0
    if fixed_pitch:
        flags |= 1 << 0
    if 0:
        flags |= 1 << 1
    if 1:
        flags |= 1 << 2
    else:
        flags |= 1 << 5
    if italic_angle:
        flags |= 1 << 6
    if 0:
        flags |= 1 << 16
    if 0:
        flags |= 1 << 17
    if 0:
        flags |= 1 << 18
    ft2font = get_font(fontfile)
    descriptor = {'Type': Name('FontDescriptor'), 'FontName': Name(t1font.prop['FontName']), 'Flags': flags, 'FontBBox': ft2font.bbox, 'ItalicAngle': italic_angle, 'Ascent': ft2font.ascender, 'Descent': ft2font.descender, 'CapHeight': 1000, 'XHeight': 500, 'FontFile': fontfileObject, 'FontFamily': t1font.prop['FamilyName'], 'StemV': 50}
    self.writeObject(fontdescObject, descriptor)
    self.outputStream(fontfileObject, b''.join(t1font.parts[:2]), extra={'Length1': len(t1font.parts[0]), 'Length2': len(t1font.parts[1]), 'Length3': 0})
    return fontdescObject