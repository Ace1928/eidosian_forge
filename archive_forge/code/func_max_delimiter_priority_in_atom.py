from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def max_delimiter_priority_in_atom(node: LN) -> Priority:
    """Return maximum delimiter priority inside `node`.

    This is specific to atoms with contents contained in a pair of parentheses.
    If `node` isn't an atom or there are no enclosing parentheses, returns 0.
    """
    if node.type != syms.atom:
        return 0
    first = node.children[0]
    last = node.children[-1]
    if not (first.type == token.LPAR and last.type == token.RPAR):
        return 0
    bt = BracketTracker()
    for c in node.children[1:-1]:
        if isinstance(c, Leaf):
            bt.mark(c)
        else:
            for leaf in c.leaves():
                bt.mark(leaf)
    try:
        return bt.max_delimiter_priority()
    except ValueError:
        return 0