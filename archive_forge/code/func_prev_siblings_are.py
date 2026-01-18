import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def prev_siblings_are(node: Optional[LN], tokens: List[Optional[NodeType]]) -> bool:
    """Return if the `node` and its previous siblings match types against the provided
    list of tokens; the provided `node`has its type matched against the last element in
    the list.  `None` can be used as the first element to declare that the start of the
    list is anchored at the start of its parent's children."""
    if not tokens:
        return True
    if tokens[-1] is None:
        return node is None
    if not node:
        return False
    if node.type != tokens[-1]:
        return False
    return prev_siblings_are(node.prev_sibling, tokens[:-1])