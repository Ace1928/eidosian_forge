import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_async_stmt_or_funcdef(leaf: Leaf) -> bool:
    """Return True if the given leaf starts an async def/for/with statement.

    Note that `async def` can be either an `async_stmt` or `async_funcdef`,
    the latter is used when it has decorators.
    """
    return bool(leaf.type == token.ASYNC and leaf.parent and (leaf.parent.type in {syms.async_stmt, syms.async_funcdef}))