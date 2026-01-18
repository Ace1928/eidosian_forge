import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_string_set(source, info):
    """Parses a string set reference."""
    source.expect('<')
    name = parse_name(source, True)
    source.expect('>')
    if name is None or name not in info.kwargs:
        raise error('undefined named list', source.string, source.pos)
    return make_string_set(info, name)