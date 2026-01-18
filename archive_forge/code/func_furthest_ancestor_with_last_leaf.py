import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def furthest_ancestor_with_last_leaf(leaf: Leaf) -> LN:
    """Returns the furthest ancestor that has this leaf node as the last leaf."""
    node: LN = leaf
    while node.parent and node.parent.children and (node is node.parent.children[-1]):
        node = node.parent
    return node