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
def chunk_IDAT(self, pos, length):
    if 'bbox' in self.im_info:
        tile = [('zip', self.im_info['bbox'], pos, self.im_rawmode)]
    else:
        if self.im_n_frames is not None:
            self.im_info['default_image'] = True
        tile = [('zip', (0, 0) + self.im_size, pos, self.im_rawmode)]
    self.im_tile = tile
    self.im_idat = length
    msg = 'image data found'
    raise EOFError(msg)