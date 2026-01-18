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
def chunk_fdAT(self, pos, length):
    if length < 4:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            s = ImageFile._safe_read(self.fp, length)
            return s
        msg = 'APNG contains truncated fDAT chunk'
        raise ValueError(msg)
    s = ImageFile._safe_read(self.fp, 4)
    seq = i32(s)
    if self._seq_num != seq - 1:
        msg = 'APNG contains frame sequence errors'
        raise SyntaxError(msg)
    self._seq_num = seq
    return self.chunk_IDAT(pos + 4, length - 4)