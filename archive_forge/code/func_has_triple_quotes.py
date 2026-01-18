import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def has_triple_quotes(string: str) -> bool:
    """
    Returns:
        True iff @string starts with three quotation characters.
    """
    raw_string = string.lstrip(STRING_PREFIX_CHARS)
    return raw_string[:3] in {'"""', "'''"}