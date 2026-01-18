from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
def stack_copy(stack: List[Tuple[DFAS, int, RawNode]]) -> List[Tuple[DFAS, int, RawNode]]:
    """Nodeless stack copy."""
    return [(dfa, label, DUMMY_NODE) for dfa, label, _ in stack]