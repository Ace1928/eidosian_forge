from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def draw_corners(pieslice):
    if full_x:
        parts = (((x0, y0, x0 + d, y0 + d), 180, 360), ((x0, y1 - d, x0 + d, y1), 0, 180))
    elif full_y:
        parts = (((x0, y0, x0 + d, y0 + d), 90, 270), ((x1 - d, y0, x1, y0 + d), 270, 90))
    else:
        parts = []
        for i, part in enumerate((((x0, y0, x0 + d, y0 + d), 180, 270), ((x1 - d, y0, x1, y0 + d), 270, 360), ((x1 - d, y1 - d, x1, y1), 0, 90), ((x0, y1 - d, x0 + d, y1), 90, 180))):
            if corners[i]:
                parts.append(part)
    for part in parts:
        if pieslice:
            self.draw.draw_pieslice(*part + (fill, 1))
        else:
            self.draw.draw_arc(*part + (ink, width))