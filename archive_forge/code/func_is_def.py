import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@property
def is_def(self) -> bool:
    """Is this a function definition? (Also returns True for async defs.)"""
    try:
        first_leaf = self.leaves[0]
    except IndexError:
        return False
    try:
        second_leaf: Optional[Leaf] = self.leaves[1]
    except IndexError:
        second_leaf = None
    return first_leaf.type == token.NAME and first_leaf.value == 'def' or (first_leaf.type == token.ASYNC and second_leaf is not None and (second_leaf.type == token.NAME) and (second_leaf.value == 'def'))