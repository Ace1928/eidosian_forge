import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_subpattern(source, info, flags_on, flags_off):
    """Parses a subpattern with scoped flags."""
    saved_flags = info.flags
    info.flags = (info.flags | flags_on) & ~flags_off
    source.ignore_space = bool(info.flags & VERBOSE)
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    return subpattern