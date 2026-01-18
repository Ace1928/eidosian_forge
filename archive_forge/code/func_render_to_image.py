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
def render_to_image(self, text, width=10000, height=80):
    """This process takes Pyglet out of the equation and uses only DirectWrite to shape and render text.
        This may allow more accurate fonts (bidi, rtl, etc) in very special circumstances at the cost of
        additional texture space.

        :Parameters:
            `text` : str
                String of text to render.

        :rtype: `ImageData`
        :return: An image of the text.
        """
    if not self._glyph_renderer:
        self._glyph_renderer = self.glyph_renderer_class(self)
    return self._glyph_renderer.render_to_image(text, width, height)