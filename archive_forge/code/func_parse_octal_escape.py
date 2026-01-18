import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_octal_escape(source, info, digits, in_set):
    """Parses an octal escape sequence."""
    saved_pos = source.pos
    ch = source.get()
    while len(digits) < 3 and ch in OCT_DIGITS:
        digits.append(ch)
        saved_pos = source.pos
        ch = source.get()
    source.pos = saved_pos
    try:
        value = int(''.join(digits), 8)
        return make_character(info, value, in_set)
    except ValueError:
        if digits[0] in OCT_DIGITS:
            raise error('incomplete escape \\%s' % ''.join(digits), source.string, source.pos)
        else:
            raise error('bad escape \\%s' % digits[0], source.string, source.pos)