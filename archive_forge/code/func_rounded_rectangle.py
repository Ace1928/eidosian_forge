from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1, *, corners=None):
    """Draw a rounded rectangle."""
    if isinstance(xy[0], (list, tuple)):
        (x0, y0), (x1, y1) = xy
    else:
        x0, y0, x1, y1 = xy
    if x1 < x0:
        msg = 'x1 must be greater than or equal to x0'
        raise ValueError(msg)
    if y1 < y0:
        msg = 'y1 must be greater than or equal to y0'
        raise ValueError(msg)
    if corners is None:
        corners = (True, True, True, True)
    d = radius * 2
    full_x, full_y = (False, False)
    if all(corners):
        full_x = d >= x1 - x0 - 1
        if full_x:
            d = x1 - x0
        full_y = d >= y1 - y0 - 1
        if full_y:
            d = y1 - y0
        if full_x and full_y:
            return self.ellipse(xy, fill, outline, width)
    if d == 0 or not any(corners):
        return self.rectangle(xy, fill, outline, width)
    r = d // 2
    ink, fill = self._getink(outline, fill)

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
    if fill is not None:
        draw_corners(True)
        if full_x:
            self.draw.draw_rectangle((x0, y0 + r + 1, x1, y1 - r - 1), fill, 1)
        else:
            self.draw.draw_rectangle((x0 + r + 1, y0, x1 - r - 1, y1), fill, 1)
        if not full_x and (not full_y):
            left = [x0, y0, x0 + r, y1]
            if corners[0]:
                left[1] += r + 1
            if corners[3]:
                left[3] -= r + 1
            self.draw.draw_rectangle(left, fill, 1)
            right = [x1 - r, y0, x1, y1]
            if corners[1]:
                right[1] += r + 1
            if corners[2]:
                right[3] -= r + 1
            self.draw.draw_rectangle(right, fill, 1)
    if ink is not None and ink != fill and (width != 0):
        draw_corners(False)
        if not full_x:
            top = [x0, y0, x1, y0 + width - 1]
            if corners[0]:
                top[0] += r + 1
            if corners[1]:
                top[2] -= r + 1
            self.draw.draw_rectangle(top, ink, 1)
            bottom = [x0, y1 - width + 1, x1, y1]
            if corners[3]:
                bottom[0] += r + 1
            if corners[2]:
                bottom[2] -= r + 1
            self.draw.draw_rectangle(bottom, ink, 1)
        if not full_y:
            left = [x0, y0, x0 + width - 1, y1]
            if corners[0]:
                left[1] += r + 1
            if corners[3]:
                left[3] -= r + 1
            self.draw.draw_rectangle(left, ink, 1)
            right = [x1 - width + 1, y0, x1, y1]
            if corners[1]:
                right[1] += r + 1
            if corners[2]:
                right[3] -= r + 1
            self.draw.draw_rectangle(right, ink, 1)