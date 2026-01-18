import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def sub_twice(regex: Pattern[str], replacement: str, original: str) -> str:
    """Replace `regex` with `replacement` twice on `original`.

    This is used by string normalization to perform replaces on
    overlapping matches.
    """
    return regex.sub(replacement, regex.sub(replacement, original))