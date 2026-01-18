import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def child_towards(ancestor: Node, descendant: LN) -> Optional[LN]:
    """Return the child of `ancestor` that contains `descendant`."""
    node: Optional[LN] = descendant
    while node and node.parent != ancestor:
        node = node.parent
    return node