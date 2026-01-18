from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def align_layout(self, text: str | bytes, width: int, segs: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]], wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode, align: Literal['left', 'center', 'right'] | Align) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
    """Convert the layout segments to an aligned layout."""
    out = []
    for lines in segs:
        sc = line_width(lines)
        if sc == width or align == 'left':
            out.append(lines)
            continue
        if align == 'right':
            out.append([(width - sc, None), *lines])
            continue
        if align != 'center':
            raise ValueError(align)
        pad_trim_left = (width - sc + 1) // 2
        out.append([(pad_trim_left, None), *lines] if pad_trim_left else lines)
    return out