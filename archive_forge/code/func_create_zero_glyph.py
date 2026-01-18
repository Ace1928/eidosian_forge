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
def create_zero_glyph(self):
    """Zero glyph is a 1x1 image that has a -1 advance. This is to fill in for ligature substitutions since
        font system requires 1 glyph per character in a string."""
    self._create_bitmap(1, 1)
    image = wic_decoder.get_image(self._bitmap)
    glyph = self.font.create_glyph(image)
    glyph.set_bearings(-self.font.descent, 0, -1)
    return glyph