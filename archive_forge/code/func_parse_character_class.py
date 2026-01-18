import re
from sys import maxunicode
from ..helpers import OCCURRENCE_INDICATORS
from .unicode_subsets import RegexError, UnicodeSubset, unicode_subset
from .character_classes import I_SHORTCUT_REPLACE, C_SHORTCUT_REPLACE, CharacterClass
def parse_character_class() -> CharacterClass:
    nonlocal pos
    nonlocal msg
    pos += 1
    if pattern[pos] == '^':
        pos += 1
        negative = True
    else:
        negative = False
    char_class_pos = pos
    while True:
        if pattern[pos] == '[':
            msg = "invalid character '[' at position {}: {!r}"
            raise RegexError(msg.format(pos, pattern))
        elif pattern[pos] == '\\':
            if pattern[pos + 1].isdigit():
                msg = 'illegal back-reference in character class at position {}: {!r}'
                raise RegexError(msg.format(pos, pattern))
            pos += 2
        elif pattern[pos] == ']' or pattern[pos:pos + 2] == '-[':
            if pos == char_class_pos:
                msg = 'empty character class at position {}: {!r}'
                raise RegexError(msg.format(pos, pattern))
            char_class_pattern = pattern[char_class_pos:pos]
            if HYPHENS_PATTERN.search(char_class_pattern) and pos - char_class_pos > 2:
                msg = "invalid character range '--' at position {}: {!r}"
                raise RegexError(msg.format(pos, pattern))
            if xsd_version == '1.0':
                hyphen_match = INVALID_HYPHEN_PATTERN.search(char_class_pattern)
                if hyphen_match is not None:
                    hyphen_pos = char_class_pos + hyphen_match.span()[1] - 2
                    msg = "unescaped character '-' at position {}: {!r}"
                    raise RegexError(msg.format(hyphen_pos, pattern))
            char_class = CharacterClass(char_class_pattern, xsd_version)
            if negative:
                char_class.complement()
            break
        else:
            pos += 1
    if pattern[pos] != ']':
        pos += 1
        subtracted_class = parse_character_class()
        pos += 1
        char_class -= subtracted_class
    return char_class