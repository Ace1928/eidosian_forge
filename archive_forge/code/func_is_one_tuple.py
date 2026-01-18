import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_one_tuple(node: LN) -> bool:
    """Return True if `node` holds a tuple with one element, with or without parens."""
    if node.type == syms.atom:
        gexp = unwrap_singleton_parenthesis(node)
        if gexp is None or gexp.type != syms.testlist_gexp:
            return False
        return len(gexp.children) == 2 and gexp.children[1].type == token.COMMA
    return node.type in IMPLICIT_TUPLE and len(node.children) == 2 and (node.children[1].type == token.COMMA)