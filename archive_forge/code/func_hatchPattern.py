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
def hatchPattern(self, hatch_style):
    if hatch_style is not None:
        edge, face, hatch = hatch_style
        if edge is not None:
            edge = tuple(edge)
        if face is not None:
            face = tuple(face)
        hatch_style = (edge, face, hatch)
    pattern = self.hatchPatterns.get(hatch_style, None)
    if pattern is not None:
        return pattern
    name = next(self._hatch_pattern_seq)
    self.hatchPatterns[hatch_style] = name
    return name