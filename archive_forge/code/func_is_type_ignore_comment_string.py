import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_type_ignore_comment_string(value: str) -> bool:
    """Return True if the given string match with type comment with
    ignore annotation."""
    return value.startswith('# type: ignore')