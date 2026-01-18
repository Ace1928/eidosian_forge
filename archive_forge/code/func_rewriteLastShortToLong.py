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
def rewriteLastShortToLong(self, value):
    self.f.seek(-2, os.SEEK_CUR)
    bytes_written = self.f.write(struct.pack(self.longFmt, value))
    if bytes_written is not None and bytes_written != 4:
        msg = f'wrote only {bytes_written} bytes but wanted 4'
        raise RuntimeError(msg)