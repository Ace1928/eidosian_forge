from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
def make_gamma_lut(exp):
    return [int((i / 255.0) ** exp * 255.0 + 0.5) for i in range(256)]