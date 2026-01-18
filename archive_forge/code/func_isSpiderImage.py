from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def isSpiderImage(filename):
    with open(filename, 'rb') as fp:
        f = fp.read(92)
    t = struct.unpack('>23f', f)
    hdrlen = isSpiderHeader(t)
    if hdrlen == 0:
        t = struct.unpack('<23f', f)
        hdrlen = isSpiderHeader(t)
    return hdrlen