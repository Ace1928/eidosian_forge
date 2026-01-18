import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def lines_with_leading_tabs_expanded(s: str) -> List[str]:
    """
    Splits string into lines and expands only leading tabs (following the normal
    Python rules)
    """
    lines = []
    for line in s.splitlines():
        stripped_line = line.lstrip()
        if not stripped_line or stripped_line == line:
            lines.append(line)
        else:
            prefix_length = len(line) - len(stripped_line)
            prefix = line[:prefix_length].expandtabs()
            lines.append(prefix + stripped_line)
    if s.endswith('\n'):
        lines.append('')
    return lines