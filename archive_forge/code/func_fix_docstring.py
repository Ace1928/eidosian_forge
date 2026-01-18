import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def fix_docstring(docstring: str, prefix: str) -> str:
    if not docstring:
        return ''
    lines = lines_with_leading_tabs_expanded(docstring)
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        last_line_idx = len(lines) - 2
        for i, line in enumerate(lines[1:]):
            stripped_line = line[indent:].rstrip()
            if stripped_line or i == last_line_idx:
                trimmed.append(prefix + stripped_line)
            else:
                trimmed.append('')
    return '\n'.join(trimmed)