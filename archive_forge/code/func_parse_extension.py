import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_extension(source, info):
    """Parses a Python extension."""
    saved_pos = source.pos
    ch = source.get()
    if ch == '<':
        name = parse_name(source)
        group = info.open_group(name)
        source.expect('>')
        saved_flags = info.flags
        try:
            subpattern = _parse_pattern(source, info)
            source.expect(')')
        finally:
            info.flags = saved_flags
            source.ignore_space = bool(info.flags & VERBOSE)
        info.close_group()
        return Group(info, group, subpattern)
    if ch == '=':
        name = parse_name(source, allow_numeric=True)
        source.expect(')')
        if info.is_open_group(name):
            raise error('cannot refer to an open group', source.string, saved_pos)
        return make_ref_group(info, name, saved_pos)
    if ch == '>' or ch == '&':
        return parse_call_named_group(source, info, saved_pos)
    source.pos = saved_pos
    raise error('unknown extension', source.string, saved_pos)