from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
def chunk_cHRM(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    raw_vals = struct.unpack('>%dI' % (len(s) // 4), s)
    self.im_info['chromaticity'] = tuple((elt / 100000.0 for elt in raw_vals))
    return s