from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
def make_linear_lut(black, white):
    if black == 0:
        return [white * i // 255 for i in range(256)]
    msg = 'unavailable when black is non-zero'
    raise NotImplementedError(msg)