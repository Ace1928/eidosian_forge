import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def make_simple_prefix(nl_count: int, form_feed: bool, empty_line: str='\n') -> str:
    """Generate a normalized prefix string."""
    if form_feed:
        return empty_line * (nl_count - 1) + '\x0c' + empty_line
    return empty_line * nl_count