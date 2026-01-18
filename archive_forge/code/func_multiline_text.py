from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def multiline_text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, *, font_size=None):
    if direction == 'ttb':
        msg = 'ttb direction is unsupported for multiline text'
        raise ValueError(msg)
    if anchor is None:
        anchor = 'la'
    elif len(anchor) != 2:
        msg = 'anchor must be a 2 character string'
        raise ValueError(msg)
    elif anchor[1] in 'tb':
        msg = 'anchor not supported for multiline text'
        raise ValueError(msg)
    if font is None:
        font = self._getfont(font_size)
    widths = []
    max_width = 0
    lines = self._multiline_split(text)
    line_spacing = self._multiline_spacing(font, spacing, stroke_width)
    for line in lines:
        line_width = self.textlength(line, font, direction=direction, features=features, language=language)
        widths.append(line_width)
        max_width = max(max_width, line_width)
    top = xy[1]
    if anchor[1] == 'm':
        top -= (len(lines) - 1) * line_spacing / 2.0
    elif anchor[1] == 'd':
        top -= (len(lines) - 1) * line_spacing
    for idx, line in enumerate(lines):
        left = xy[0]
        width_difference = max_width - widths[idx]
        if anchor[0] == 'm':
            left -= width_difference / 2.0
        elif anchor[0] == 'r':
            left -= width_difference
        if align == 'left':
            pass
        elif align == 'center':
            left += width_difference / 2.0
        elif align == 'right':
            left += width_difference
        else:
            msg = 'align must be "left", "center" or "right"'
            raise ValueError(msg)
        self.text((left, top), line, fill, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width, stroke_fill=stroke_fill, embedded_color=embedded_color)
        top += line_spacing