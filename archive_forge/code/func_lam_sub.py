from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
def lam_sub(grammar: Grammar, node: RawNode) -> NL:
    assert node[3] is not None
    return Node(type=node[0], children=node[3], context=node[2])