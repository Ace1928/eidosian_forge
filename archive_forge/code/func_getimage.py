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
def getimage(self, size, bpp=False):
    """
        Get an image from the icon
        """
    return self.frame(self.getentryindex(size, bpp))