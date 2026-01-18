import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def get_annotation_type(leaf: Leaf) -> Literal['return', 'param', None]:
    """Returns the type of annotation this leaf is part of, if any."""
    ancestor = leaf.parent
    while ancestor is not None:
        if ancestor.prev_sibling and ancestor.prev_sibling.type == token.RARROW:
            return 'return'
        if ancestor.parent and ancestor.parent.type == syms.tname:
            return 'param'
        ancestor = ancestor.parent
    return None