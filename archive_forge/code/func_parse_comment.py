import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_comment(source):
    """Parses a comment."""
    while True:
        saved_pos = source.pos
        c = source.get(True)
        if not c or c == ')':
            break
        if c == '\\':
            c = source.get(True)
    source.pos = saved_pos
    source.expect(')')
    return None