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
def sizes(self):
    """
        Get a list of all available icon sizes and color depths.
        """
    return {(h['width'], h['height']) for h in self.entry}