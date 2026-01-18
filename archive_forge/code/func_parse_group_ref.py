import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_group_ref(source, info):
    """Parses a group reference."""
    source.expect('<')
    saved_pos = source.pos
    name = parse_name(source, True)
    source.expect('>')
    if info.is_open_group(name):
        raise error('cannot refer to an open group', source.string, source.pos)
    return make_ref_group(info, name, saved_pos)