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
def chunk_fcTL(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    if length < 26:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            return s
        msg = 'APNG contains truncated fcTL chunk'
        raise ValueError(msg)
    seq = i32(s)
    if self._seq_num is None and seq != 0 or (self._seq_num is not None and self._seq_num != seq - 1):
        msg = 'APNG contains frame sequence errors'
        raise SyntaxError(msg)
    self._seq_num = seq
    width, height = (i32(s, 4), i32(s, 8))
    px, py = (i32(s, 12), i32(s, 16))
    im_w, im_h = self.im_size
    if px + width > im_w or py + height > im_h:
        msg = 'APNG contains invalid frames'
        raise SyntaxError(msg)
    self.im_info['bbox'] = (px, py, px + width, py + height)
    delay_num, delay_den = (i16(s, 20), i16(s, 22))
    if delay_den == 0:
        delay_den = 100
    self.im_info['duration'] = float(delay_num) / float(delay_den) * 1000
    self.im_info['disposal'] = s[24]
    self.im_info['blend'] = s[25]
    return s