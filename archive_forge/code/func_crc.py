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
def crc(self, cid, data):
    """Read and verify checksum"""
    if ImageFile.LOAD_TRUNCATED_IMAGES and cid[0] >> 5 & 1:
        self.crc_skip(cid, data)
        return
    try:
        crc1 = _crc32(data, _crc32(cid))
        crc2 = i32(self.fp.read(4))
        if crc1 != crc2:
            msg = f'broken PNG file (bad header checksum in {repr(cid)})'
            raise SyntaxError(msg)
    except struct.error as e:
        msg = f'broken PNG file (incomplete checksum in {repr(cid)})'
        raise SyntaxError(msg) from e