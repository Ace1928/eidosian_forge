from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def coord_at_angle(coord, angle):
    x, y = coord
    angle -= 90
    distance = width / 2 - 1
    return tuple((p + (math.floor(p_d) if p_d > 0 else math.ceil(p_d)) for p, p_d in ((x, distance * math.cos(math.radians(angle))), (y, distance * math.sin(math.radians(angle))))))