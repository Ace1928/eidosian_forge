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
def getchunks(im, **params):
    """Return a list of PNG chunks representing this image."""

    class collector:
        data = []

        def write(self, data):
            pass

        def append(self, chunk):
            self.data.append(chunk)

    def append(fp, cid, *data):
        data = b''.join(data)
        crc = o32(_crc32(data, _crc32(cid)))
        fp.append((cid, data, crc))
    fp = collector()
    try:
        im.encoderinfo = params
        _save(im, fp, None, append)
    finally:
        del im.encoderinfo
    return fp.data