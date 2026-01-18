import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_escape(source, info, in_set):
    """Parses an escape sequence."""
    saved_ignore = source.ignore_space
    source.ignore_space = False
    ch = source.get()
    source.ignore_space = saved_ignore
    if not ch:
        raise error('bad escape (end of pattern)', source.string, source.pos)
    if ch in HEX_ESCAPES:
        return parse_hex_escape(source, info, ch, HEX_ESCAPES[ch], in_set, ch)
    elif ch == 'g' and (not in_set):
        saved_pos = source.pos
        try:
            return parse_group_ref(source, info)
        except error:
            source.pos = saved_pos
        return make_character(info, ord(ch), in_set)
    elif ch == 'G' and (not in_set):
        return SearchAnchor()
    elif ch == 'L' and (not in_set):
        return parse_string_set(source, info)
    elif ch == 'N':
        return parse_named_char(source, info, in_set)
    elif ch in 'pP':
        return parse_property(source, info, ch == 'p', in_set)
    elif ch == 'R' and (not in_set):
        charset = [10, 11, 12, 13]
        if info.guess_encoding == UNICODE:
            charset.extend([133, 8232, 8233])
        return Atomic(Branch([String([13, 10]), SetUnion(info, [Character(c) for c in charset])]))
    elif ch == 'X' and (not in_set):
        return Grapheme()
    elif ch in ALPHA:
        if not in_set:
            if info.flags & WORD:
                value = WORD_POSITION_ESCAPES.get(ch)
            else:
                value = POSITION_ESCAPES.get(ch)
            if value:
                return value
        value = CHARSET_ESCAPES.get(ch)
        if value:
            return value
        value = CHARACTER_ESCAPES.get(ch)
        if value:
            return Character(ord(value))
        raise error('bad escape \\%s' % ch, source.string, source.pos)
    elif ch in DIGITS:
        return parse_numeric_escape(source, info, ch, in_set)
    else:
        return make_character(info, ord(ch), in_set)