import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_tuple_containing_walrus(node: LN) -> bool:
    """Return True if `node` holds a tuple that contains a walrus operator."""
    if node.type != syms.atom:
        return False
    gexp = unwrap_singleton_parenthesis(node)
    if gexp is None or gexp.type != syms.testlist_gexp:
        return False
    return any((child.type == syms.namedexpr_test for child in gexp.children))