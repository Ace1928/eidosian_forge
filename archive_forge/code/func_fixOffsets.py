from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
def fixOffsets(self, count, isShort=False, isLong=False):
    if not isShort and (not isLong):
        msg = 'offset is neither short nor long'
        raise RuntimeError(msg)
    for i in range(count):
        offset = self.readShort() if isShort else self.readLong()
        offset += self.offsetOfNewPage
        if isShort and offset >= 65536:
            if count != 1:
                msg = 'not implemented'
                raise RuntimeError(msg)
            self.rewriteLastShortToLong(offset)
            self.f.seek(-10, os.SEEK_CUR)
            self.writeShort(TiffTags.LONG)
            self.f.seek(8, os.SEEK_CUR)
        elif isShort:
            self.rewriteLastShort(offset)
        else:
            self.rewriteLastLong(offset)