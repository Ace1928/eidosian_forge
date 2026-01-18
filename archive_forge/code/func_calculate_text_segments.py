from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def calculate_text_segments(self, text: str | bytes, width: int, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
    """
        Calculate the segments of text to display given width screen columns to display them.

        text - unicode text or byte string to display
        width - number of available screen columns
        wrap - wrapping mode used

        Returns a layout structure without an alignment applied.
        """
    if wrap in {'clip', 'ellipsis'}:
        return self._calculate_trimmed_segments(text, width, wrap)
    nl, nl_o, sp_o = ('\n', '\n', ' ')
    if isinstance(text, bytes):
        nl = b'\n'
        nl_o = ord(nl_o)
        sp_o = ord(sp_o)
    segments = []
    idx = 0
    while idx <= len(text):
        nl_pos = text.find(nl, idx)
        if nl_pos == -1:
            nl_pos = len(text)
        screen_columns = calc_width(text, idx, nl_pos)
        if screen_columns == 0:
            segments.append([(0, nl_pos)])
            idx = nl_pos + 1
            continue
        if screen_columns <= width:
            segments.append([(screen_columns, idx, nl_pos), (0, nl_pos)])
            idx = nl_pos + 1
            continue
        pos, screen_columns = calc_text_pos(text, idx, nl_pos, width)
        if pos == idx:
            raise CanNotDisplayText('Wide character will not fit in 1-column width')
        if wrap == 'any':
            segments.append([(screen_columns, idx, pos)])
            idx = pos
            continue
        if wrap != 'space':
            raise ValueError(wrap)
        if text[pos] == sp_o:
            segments.append([(screen_columns, idx, pos), (0, pos)])
            idx = pos + 1
            continue
        if is_wide_char(text, pos):
            segments.append([(screen_columns, idx, pos)])
            idx = pos
            continue
        prev = pos
        while prev > idx:
            prev = move_prev_char(text, idx, prev)
            if text[prev] == sp_o:
                screen_columns = calc_width(text, idx, prev)
                line = [(0, prev)]
                if idx != prev:
                    line = [(screen_columns, idx, prev), *line]
                segments.append(line)
                idx = prev + 1
                break
            if is_wide_char(text, prev):
                next_char = move_next_char(text, prev, pos)
                screen_columns = calc_width(text, idx, next_char)
                segments.append([(screen_columns, idx, next_char)])
                idx = next_char
                break
        else:
            if segments and (len(segments[-1]) == 2 or (len(segments[-1]) == 1 and len(segments[-1][0]) == 2)):
                if len(segments[-1]) == 1:
                    [(h_sc, h_off)] = segments[-1]
                    p_sc = 0
                    p_off = _p_end = h_off
                else:
                    [(p_sc, p_off, _p_end), (h_sc, h_off)] = segments[-1]
                if p_sc < width and h_sc == 0 and (text[h_off] == sp_o):
                    del segments[-1]
                    idx = p_off
                    pos, screen_columns = calc_text_pos(text, idx, nl_pos, width)
                    segments.append([(screen_columns, idx, pos)])
                    idx = pos
                    if idx < len(text) and text[idx] in {sp_o, nl_o}:
                        segments[-1].append((0, idx))
                        idx += 1
                    continue
            segments.append([(screen_columns, idx, pos)])
            idx = pos
    return segments