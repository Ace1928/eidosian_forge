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
def chunk_sRGB(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    if length < 1:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            return s
        msg = 'Truncated sRGB chunk'
        raise ValueError(msg)
    self.im_info['srgb'] = s[0]
    return s