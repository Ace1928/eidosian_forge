from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def subseg(self, text: str | bytes, start: int, end: int) -> list[tuple[int, int] | tuple[int, int, int | bytes]]:
    """
        Return a "sub-segment" list containing segment structures
        that make up a portion of this segment.

        A list is returned to handle cases where wide characters
        need to be replaced with a space character at either edge
        so two or three segments will be returned.
        """
    start = max(start, 0)
    end = min(end, self.sc)
    if start >= end:
        return []
    if self.text:
        spos, epos, pad_left, pad_right = calc_trim_text(self.text, 0, len(self.text), start, end)
        return [(end - start, self.offs, b''.ljust(pad_left) + self.text[spos:epos] + b''.ljust(pad_right))]
    if self.end:
        spos, epos, pad_left, pad_right = calc_trim_text(text, self.offs, self.end, start, end)
        lines = []
        if pad_left:
            lines.append((1, spos - 1))
        lines.append((end - start - pad_left - pad_right, spos, epos))
        if pad_right:
            lines.append((1, epos))
        return lines
    return [(end - start, self.offs)]