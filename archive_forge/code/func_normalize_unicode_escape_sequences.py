import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def normalize_unicode_escape_sequences(leaf: Leaf) -> None:
    """Replace hex codes in Unicode escape sequences with lowercase representation."""
    text = leaf.value
    prefix = get_string_prefix(text)
    if 'r' in prefix.lower():
        return

    def replace(m: Match[str]) -> str:
        groups = m.groupdict()
        back_slashes = groups['backslashes']
        if len(back_slashes) % 2 == 0:
            return back_slashes + groups['body']
        if groups['u']:
            return back_slashes + 'u' + groups['u'].lower()
        elif groups['U']:
            return back_slashes + 'U' + groups['U'].lower()
        elif groups['x']:
            return back_slashes + 'x' + groups['x'].lower()
        else:
            assert groups['N'], f'Unexpected match: {m}'
            return back_slashes + 'N{' + groups['N'].upper() + '}'
    leaf.value = re.sub(UNICODE_ESCAPE_RE, replace, text)