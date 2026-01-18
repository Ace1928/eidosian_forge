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
def skipIFDs(self):
    while True:
        ifd_offset = self.readLong()
        if ifd_offset == 0:
            self.whereToWriteNewIFDOffset = self.f.tell() - 4
            break
        self.f.seek(ifd_offset)
        num_tags = self.readShort()
        self.f.seek(num_tags * 12, os.SEEK_CUR)