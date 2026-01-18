from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def dataforsize(self, size):
    """
        Get an icon resource as {channel: array}.  Note that
        the arrays are bottom-up like windows bitmaps and will likely
        need to be flipped or transposed in some way.
        """
    dct = {}
    for code, reader in self.SIZES[size]:
        desc = self.dct.get(code)
        if desc is not None:
            dct.update(reader(self.fobj, desc, size))
    return dct