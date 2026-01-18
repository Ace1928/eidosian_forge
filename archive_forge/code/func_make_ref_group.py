import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_ref_group(info, name, position):
    """Makes a group reference."""
    return RefGroup(info, name, position, case_flags=make_case_flags(info))