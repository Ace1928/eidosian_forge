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
def goToEnd(self):
    self.f.seek(0, os.SEEK_END)
    pos = self.f.tell()
    pad_bytes = 16 - pos % 16
    if 0 < pad_bytes < 16:
        self.f.write(bytes(pad_bytes))
    self.offsetOfNewPage = self.f.tell()