import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_count(source):
    """Parses a quantifier's count, which can be empty."""
    return source.get_while(DIGITS)