from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def read_png_or_jpeg2000(fobj, start_length, size):
    start, length = start_length
    fobj.seek(start)
    sig = fobj.read(12)
    if sig[:8] == b'\x89PNG\r\n\x1a\n':
        fobj.seek(start)
        im = PngImagePlugin.PngImageFile(fobj)
        Image._decompression_bomb_check(im.size)
        return {'RGBA': im}
    elif sig[:4] == b'\xffO\xffQ' or sig[:4] == b'\r\n\x87\n' or sig == b'\x00\x00\x00\x0cjP  \r\n\x87\n':
        if not enable_jpeg2k:
            msg = 'Unsupported icon subimage format (rebuild PIL with JPEG 2000 support to fix this)'
            raise ValueError(msg)
        fobj.seek(start)
        jp2kstream = fobj.read(length)
        f = io.BytesIO(jp2kstream)
        im = Jpeg2KImagePlugin.Jpeg2KImageFile(f)
        Image._decompression_bomb_check(im.size)
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        return {'RGBA': im}
    else:
        msg = 'Unsupported icon subimage format'
        raise ValueError(msg)