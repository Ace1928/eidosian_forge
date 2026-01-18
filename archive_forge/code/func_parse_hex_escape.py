import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_hex_escape(source, info, esc, expected_len, in_set, type):
    """Parses a hex escape sequence."""
    saved_pos = source.pos
    digits = []
    for i in range(expected_len):
        ch = source.get()
        if ch not in HEX_DIGITS:
            raise error('incomplete escape \\%s%s' % (type, ''.join(digits)), source.string, saved_pos)
        digits.append(ch)
    try:
        value = int(''.join(digits), 16)
    except ValueError:
        pass
    else:
        if value < 1114112:
            return make_character(info, value, in_set)
    raise error('bad hex escape \\%s%s' % (esc, ''.join(digits)), source.string, saved_pos)