import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_repl_named_char(source):
    """Parses a named character in a replacement string."""
    saved_pos = source.pos
    if source.match('{'):
        name = source.get_while(ALPHA | set(' '))
        if source.match('}'):
            try:
                value = unicodedata.lookup(name)
                return ord(value)
            except KeyError:
                raise error('undefined character name', source.string, source.pos)
    source.pos = saved_pos
    return None