from __future__ import annotations
import re
from . import Image, _imagingmorph
def build_default_lut(self):
    symbols = [0, 1]
    m = 1 << 4
    self.lut = bytearray((symbols[i & m > 0] for i in range(LUT_SIZE)))