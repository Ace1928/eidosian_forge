import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def normalize_string_prefix(s: str) -> str:
    """Make all string prefixes lowercase."""
    match = STRING_PREFIX_RE.match(s)
    assert match is not None, f'failed to match string {s!r}'
    orig_prefix = match.group(1)
    new_prefix = orig_prefix.replace('F', 'f').replace('B', 'b').replace('U', '').replace('u', '')
    if len(new_prefix) == 2 and 'r' != new_prefix[0].lower():
        new_prefix = new_prefix[::-1]
    return f'{new_prefix}{match.group(2)}'