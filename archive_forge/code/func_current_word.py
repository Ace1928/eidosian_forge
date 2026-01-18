import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_word(cursor_offset: int, line: str) -> Optional[LinePart]:
    """the object.attribute.attribute just before or under the cursor"""
    start = cursor_offset
    end = cursor_offset
    word = None
    for m in _current_word_re.finditer(line):
        if m.start(1) < cursor_offset <= m.end(1):
            start = m.start(1)
            end = m.end(1)
            word = m.group(1)
    if word is None:
        return None
    return LinePart(start, end, word)