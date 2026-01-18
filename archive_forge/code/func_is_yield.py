import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_yield(node: LN) -> bool:
    """Return True if `node` holds a `yield` or `yield from` expression."""
    if node.type == syms.yield_expr:
        return True
    if is_name_token(node) and node.value == 'yield':
        return True
    if node.type != syms.atom:
        return False
    if len(node.children) != 3:
        return False
    lpar, expr, rpar = node.children
    if lpar.type == token.LPAR and rpar.type == token.RPAR:
        return is_yield(expr)
    return False