import re
from . import utilities
def parse_int_or_empty(s):
    """
    >>> parse_int_or_empty('3')
    3
    >>> parse_int_or_empty('') is None
    True
    """
    if s:
        return int(s)
    return None