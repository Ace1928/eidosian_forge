import os
import re
import sys
from typing import List
def find_matching_paren(s: str, startpos: int) -> int:
    """Given a string "prefix (unknown number of characters) suffix"
    and the position of the first `(` returns the index of the character
    1 past the `)`, accounting for paren nesting
    """
    opening = 0
    for i, c in enumerate(s[startpos:]):
        if c == '(':
            opening += 1
        elif c == ')':
            opening -= 1
            if opening == 0:
                return startpos + i + 1
    raise IndexError('Closing parens not found!')