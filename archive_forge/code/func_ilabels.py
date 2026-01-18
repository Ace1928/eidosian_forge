from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
@property
def ilabels(self) -> Set[int]:
    return self._dead_ilabels.symmetric_difference(self._ilabels)