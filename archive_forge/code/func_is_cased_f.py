import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_cased_f(flags, char):
    """Checks whether a character is cased."""
    return len(_regex.get_all_cases(flags, char)) > 1