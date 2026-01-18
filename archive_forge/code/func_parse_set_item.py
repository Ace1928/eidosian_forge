import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_set_item(source, info):
    """Parses an item in a character set."""
    version = info.flags & _ALL_VERSIONS or DEFAULT_VERSION
    if source.match('\\'):
        return parse_escape(source, info, True)
    saved_pos = source.pos
    if source.match('[:'):
        try:
            return parse_posix_class(source, info)
        except ParseError:
            source.pos = saved_pos
    if version == VERSION1 and source.match('['):
        negate = source.match('^')
        item = parse_set_union(source, info)
        if not source.match(']'):
            raise error('missing ]', source.string, source.pos)
        if negate:
            item = item.with_flags(positive=not item.positive)
        return item
    ch = source.get()
    if not ch:
        raise error('unterminated character set', source.string, source.pos)
    return Character(ord(ch))