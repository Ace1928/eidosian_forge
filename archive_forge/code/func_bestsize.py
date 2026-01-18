from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def bestsize(self):
    sizes = self.itersizes()
    if not sizes:
        msg = 'No 32bit icon resources found'
        raise SyntaxError(msg)
    return max(sizes)