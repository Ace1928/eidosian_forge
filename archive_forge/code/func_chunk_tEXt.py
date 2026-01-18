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
def chunk_tEXt(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    try:
        k, v = s.split(b'\x00', 1)
    except ValueError:
        k = s
        v = b''
    if k:
        k = k.decode('latin-1', 'strict')
        v_str = v.decode('latin-1', 'replace')
        self.im_info[k] = v if k == 'exif' else v_str
        self.im_text[k] = v_str
        self.check_text_memory(len(v_str))
    return s