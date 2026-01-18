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
def fixIFD(self):
    num_tags = self.readShort()
    for i in range(num_tags):
        tag, field_type, count = struct.unpack(self.tagFormat, self.f.read(8))
        field_size = self.fieldSizes[field_type]
        total_size = field_size * count
        is_local = total_size <= 4
        if not is_local:
            offset = self.readLong()
            offset += self.offsetOfNewPage
            self.rewriteLastLong(offset)
        if tag in self.Tags:
            cur_pos = self.f.tell()
            if is_local:
                self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                self.f.seek(cur_pos + 4)
            else:
                self.f.seek(offset)
                self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                self.f.seek(cur_pos)
            offset = cur_pos = None
        elif is_local:
            self.f.seek(4, os.SEEK_CUR)