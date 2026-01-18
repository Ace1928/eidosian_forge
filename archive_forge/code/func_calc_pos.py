from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def calc_pos(text: str | bytes, layout: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]], pref_col: Literal['left', 'right', Align.LEFT, Align.RIGHT] | int, row: int) -> int:
    """
    Calculate the closest linear position to pref_col and row given a
    layout structure.
    """
    if row < 0 or row >= len(layout):
        raise ValueError('calculate_pos: out of layout row range')
    pos = calc_line_pos(text, layout[row], pref_col)
    if pos is not None:
        return pos
    rows_above = list(range(row - 1, -1, -1))
    rows_below = list(range(row + 1, len(layout)))
    while rows_above and rows_below:
        if rows_above:
            r = rows_above.pop(0)
            pos = calc_line_pos(text, layout[r], pref_col)
            if pos is not None:
                return pos
        if rows_below:
            r = rows_below.pop(0)
            pos = calc_line_pos(text, layout[r], pref_col)
            if pos is not None:
                return pos
    return 0