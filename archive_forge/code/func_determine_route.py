from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
def determine_route(self, value: Optional[str]=None, force: bool=False) -> Optional[int]:
    alive_ilabels = self.ilabels
    if len(alive_ilabels) == 0:
        *_, most_successful_ilabel = self._dead_ilabels
        raise ParseError('bad input', most_successful_ilabel, value, self.context)
    ilabel, *rest = alive_ilabels
    if force or not rest:
        return ilabel
    else:
        return None