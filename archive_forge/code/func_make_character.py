import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_character(info, value, in_set=False):
    """Makes a character literal."""
    if in_set:
        return Character(value)
    return Character(value, case_flags=make_case_flags(info))