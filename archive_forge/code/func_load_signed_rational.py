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
@_register_loader(10, 8)
def load_signed_rational(self, data, legacy_api=True):
    vals = self._unpack(f'{len(data) // 4}l', data)

    def combine(a, b):
        return (a, b) if legacy_api else IFDRational(a, b)
    return tuple((combine(num, denom) for num, denom in zip(vals[::2], vals[1::2])))