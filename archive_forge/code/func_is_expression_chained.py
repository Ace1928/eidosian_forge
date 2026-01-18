import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def is_expression_chained(chained_leaves: List[Leaf]) -> bool:
    """
    Function to determine if the variable is a chained call.
    (e.g., foo.lookup, foo().lookup, (foo.lookup())) will be recognized as chained call)
    """
    if len(chained_leaves) < 2:
        return True
    current_leaf = chained_leaves[-1]
    past_leaf = chained_leaves[-2]
    if past_leaf.type == token.NAME:
        return current_leaf.type in {token.DOT}
    elif past_leaf.type in {token.RPAR, token.RSQB}:
        return current_leaf.type in {token.RSQB, token.RPAR}
    elif past_leaf.type in {token.LPAR, token.LSQB}:
        return current_leaf.type in {token.NAME, token.LPAR, token.LSQB}
    else:
        return False