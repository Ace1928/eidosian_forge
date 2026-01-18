from __future__ import annotations
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence, Union, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth, take_using_weights, to_int, to_str
from .controls import (
from .dimension import (
from .margins import Margin
from .mouse_handlers import MouseHandlers
from .screen import _CHAR_CACHE, Screen, WritePosition
from .utils import explode_text_fragments
def copy_line(line: StyleAndTextTuples, lineno: int, x: int, y: int, is_input: bool=False) -> tuple[int, int]:
    """
            Copy over a single line to the output screen. This can wrap over
            multiple lines in the output. It will call the prefix (prompt)
            function before every line.
            """
    if is_input:
        current_rowcol_to_yx = rowcol_to_yx
    else:
        current_rowcol_to_yx = {}
    if is_input and get_line_prefix:
        prompt = to_formatted_text(get_line_prefix(lineno, 0))
        x, y = copy_line(prompt, lineno, x, y, is_input=False)
    skipped = 0
    if horizontal_scroll and is_input:
        h_scroll = horizontal_scroll
        line = explode_text_fragments(line)
        while h_scroll > 0 and line:
            h_scroll -= get_cwidth(line[0][1])
            skipped += 1
            del line[:1]
        x -= h_scroll
    if align == WindowAlign.CENTER:
        line_width = fragment_list_width(line)
        if line_width < width:
            x += (width - line_width) // 2
    elif align == WindowAlign.RIGHT:
        line_width = fragment_list_width(line)
        if line_width < width:
            x += width - line_width
    col = 0
    wrap_count = 0
    for style, text, *_ in line:
        new_buffer_row = new_buffer[y + ypos]
        if '[ZeroWidthEscape]' in style:
            new_screen.zero_width_escapes[y + ypos][x + xpos] += text
            continue
        for c in text:
            char = _CHAR_CACHE[c, style]
            char_width = char.width
            if wrap_lines and x + char_width > width:
                visible_line_to_row_col[y + 1] = (lineno, visible_line_to_row_col[y][1] + x)
                y += 1
                wrap_count += 1
                x = 0
                if is_input and get_line_prefix:
                    prompt = to_formatted_text(get_line_prefix(lineno, wrap_count))
                    x, y = copy_line(prompt, lineno, x, y, is_input=False)
                new_buffer_row = new_buffer[y + ypos]
                if y >= write_position.height:
                    return (x, y)
            if x >= 0 and y >= 0 and (x < width):
                new_buffer_row[x + xpos] = char
                if char_width > 1:
                    for i in range(1, char_width):
                        new_buffer_row[x + xpos + i] = empty_char
                elif char_width == 0:
                    for pw in [2, 1]:
                        if x - pw >= 0 and new_buffer_row[x + xpos - pw].width == pw:
                            prev_char = new_buffer_row[x + xpos - pw]
                            char2 = _CHAR_CACHE[prev_char.char + c, prev_char.style]
                            new_buffer_row[x + xpos - pw] = char2
                current_rowcol_to_yx[lineno, col + skipped] = (y + ypos, x + xpos)
            col += 1
            x += char_width
    return (x, y)