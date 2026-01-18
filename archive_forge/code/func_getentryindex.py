from __future__ import annotations
import warnings
from io import BytesIO
from math import ceil, log
from . import BmpImagePlugin, Image, ImageFile, PngImagePlugin
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
def getentryindex(self, size, bpp=False):
    for i, h in enumerate(self.entry):
        if size == h['dim'] and (bpp is False or bpp == h['color_depth']):
            return i
    return 0