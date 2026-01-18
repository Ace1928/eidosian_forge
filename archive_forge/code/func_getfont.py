from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def getfont(self):
    """
        Get the current default font.

        To set the default font for this ImageDraw instance::

            from PIL import ImageDraw, ImageFont
            draw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        To set the default font for all future ImageDraw instances::

            from PIL import ImageDraw, ImageFont
            ImageDraw.ImageDraw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        If the current default font is ``None``,
        it is initialized with ``ImageFont.load_default()``.

        :returns: An image font."""
    if not self.font:
        from . import ImageFont
        self.font = ImageFont.load_default()
    return self.font