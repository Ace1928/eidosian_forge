import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_hexadecimal(string):
    """Checks whether a string is hexadecimal."""
    return all((ch in HEX_DIGITS for ch in string))