import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def is_color_run(self, run):
    """Will return True if the run contains a colored glyph."""
    try:
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            enumerator = IDWriteColorGlyphRunEnumerator1()
            color = self.font._write_factory.TranslateColorGlyphRun4(no_offset, run, None, DWRITE_GLYPH_IMAGE_FORMATS_ALL, self.measuring_mode, None, 0, byref(enumerator))
        elif WINDOWS_8_1_OR_GREATER:
            enumerator = IDWriteColorGlyphRunEnumerator()
            color = self.font._write_factory.TranslateColorGlyphRun(0.0, 0.0, run, None, self.measuring_mode, None, 0, byref(enumerator))
        else:
            return False
        return True
    except OSError as dw_err:
        if dw_err.winerror != -2003283956:
            raise dw_err
    return False