from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def getink(fill):
    ink, fill = self._getink(fill)
    if ink is None:
        return fill
    return ink