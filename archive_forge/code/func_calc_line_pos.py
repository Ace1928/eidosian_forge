from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def calc_line_pos(text: str | bytes, line_layout, pref_col: Literal['left', 'right', Align.LEFT, Align.RIGHT] | int):
    """
    Calculate the closest linear position to pref_col given a
    line layout structure.  Returns None if no position found.
    """
    closest_sc = None
    closest_pos = None
    current_sc = 0
    if pref_col == 'left':
        for seg in line_layout:
            s = LayoutSegment(seg)
            if s.offs is not None:
                return s.offs
        return None
    if pref_col == 'right':
        for seg in line_layout:
            s = LayoutSegment(seg)
            if s.offs is not None:
                closest_pos = s
        s = closest_pos
        if s is None:
            return None
        if s.end is None:
            return s.offs
        return calc_text_pos(text, s.offs, s.end, s.sc - 1)[0]
    for seg in line_layout:
        s = LayoutSegment(seg)
        if s.offs is not None:
            if s.end is not None:
                if current_sc <= pref_col < current_sc + s.sc:
                    return calc_text_pos(text, s.offs, s.end, pref_col - current_sc)[0]
                if current_sc <= pref_col:
                    closest_sc = current_sc + s.sc - 1
                    closest_pos = s
            if closest_sc is None or abs(pref_col - current_sc) < abs(pref_col - closest_sc):
                closest_sc = current_sc
                closest_pos = s.offs
            if current_sc > closest_sc:
                break
        current_sc += s.sc
    if closest_pos is None or isinstance(closest_pos, int):
        return closest_pos
    s = closest_pos
    return calc_text_pos(text, s.offs, s.end, s.sc - 1)[0]