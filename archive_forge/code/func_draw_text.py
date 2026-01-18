from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def draw_text(ink, stroke_width=0, stroke_offset=None):
    mode = self.fontmode
    if stroke_width == 0 and embedded_color:
        mode = 'RGBA'
    coord = []
    start = []
    for i in range(2):
        coord.append(int(xy[i]))
        start.append(math.modf(xy[i])[0])
    try:
        mask, offset = font.getmask2(text, mode, *args, direction=direction, features=features, language=language, stroke_width=stroke_width, anchor=anchor, ink=ink, start=start, **kwargs)
        coord = (coord[0] + offset[0], coord[1] + offset[1])
    except AttributeError:
        try:
            mask = font.getmask(text, mode, direction, features, language, stroke_width, anchor, ink, *args, start=start, **kwargs)
        except TypeError:
            mask = font.getmask(text)
    if stroke_offset:
        coord = (coord[0] + stroke_offset[0], coord[1] + stroke_offset[1])
    if mode == 'RGBA':
        color, mask = (mask, mask.getband(3))
        ink_alpha = struct.pack('i', ink)[3]
        color.fillband(3, ink_alpha)
        x, y = coord
        self.im.paste(color, (x, y, x + mask.size[0], y + mask.size[1]), mask)
    else:
        self.draw.draw_bitmap(coord, mask, ink)