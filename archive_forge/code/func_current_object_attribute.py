import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_object_attribute(cursor_offset: int, line: str) -> Optional[LinePart]:
    """If in attribute completion, the attribute being completed"""
    match = current_word(cursor_offset, line)
    if match is None:
        return None
    matches = _current_object_attribute_re.finditer(match.word)
    next(matches)
    for m in matches:
        if m.start(1) + match.start <= cursor_offset <= m.end(1) + match.start:
            return LinePart(m.start(1) + match.start, m.end(1) + match.start, m.group(1))
    return None