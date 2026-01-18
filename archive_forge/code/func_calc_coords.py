from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def calc_coords(text: str | bytes, layout: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]], pos: int, clamp: int=1) -> tuple[int, int]:
    """
    Calculate the coordinates closest to position pos in text with layout.

    text -- raw string or unicode string
    layout -- layout structure applied to text
    pos -- integer position into text
    clamp -- ignored right now
    """
    closest: tuple[int, tuple[int, int]] | None = None
    y = 0
    for line_layout in layout:
        x = 0
        for seg in line_layout:
            s = LayoutSegment(seg)
            if s.offs is None:
                x += s.sc
                continue
            if s.offs == pos:
                return (x, y)
            if s.end is not None and s.offs <= pos < s.end:
                x += calc_width(text, s.offs, pos)
                return (x, y)
            distance = abs(s.offs - pos)
            if s.end is not None and s.end < pos:
                distance = pos - (s.end - 1)
            if closest is None or distance < closest[0]:
                closest = (distance, (x, y))
            x += s.sc
        y += 1
    if closest:
        return closest[1]
    return (0, 0)