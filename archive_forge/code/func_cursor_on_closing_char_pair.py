import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def cursor_on_closing_char_pair(cursor_offset: int, line: str, ch: Optional[str]=None) -> Tuple[bool, bool]:
    """Checks if cursor sits on closing character of a pair
    and whether its pair character is directly behind it
    """
    on_closing_char, pair_close = (False, False)
    if line is None:
        return (on_closing_char, pair_close)
    if cursor_offset < len(line):
        cur_char = line[cursor_offset]
        if cur_char in CHARACTER_PAIR_MAP.values():
            on_closing_char = True if ch is None else cur_char == ch
        if cursor_offset > 0:
            prev_char = line[cursor_offset - 1]
            if on_closing_char and prev_char in CHARACTER_PAIR_MAP and (CHARACTER_PAIR_MAP[prev_char] == cur_char):
                pair_close = True if ch is None else prev_char == ch
    return (on_closing_char, pair_close)