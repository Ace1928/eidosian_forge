import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_limited_quantifier(source):
    """Parses a limited quantifier."""
    saved_pos = source.pos
    min_count = parse_count(source)
    if source.match(','):
        max_count = parse_count(source)
        min_count = int(min_count or 0)
        max_count = int(max_count) if max_count else None
    else:
        if not min_count:
            source.pos = saved_pos
            return None
        min_count = max_count = int(min_count)
    if not source.match('}'):
        source.pos = saved_pos
        return None
    if is_above_limit(min_count) or is_above_limit(max_count):
        raise error('repeat count too big', source.string, saved_pos)
    if max_count is not None and min_count > max_count:
        raise error('min repeat greater than max repeat', source.string, saved_pos)
    return (min_count, max_count)