from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def bitmap(self, xy, bitmap, fill=None):
    """Draw a bitmap."""
    bitmap.load()
    ink, fill = self._getink(fill)
    if ink is None:
        ink = fill
    if ink is not None:
        self.draw.draw_bitmap(xy, bitmap.im, ink)