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
def is_complex_subscript(self, leaf: Leaf) -> bool:
    """Return True iff `leaf` is part of a slice with non-trivial exprs."""
    open_lsqb = self.bracket_tracker.get_open_lsqb()
    if open_lsqb is None:
        return False
    subscript_start = open_lsqb.next_sibling
    if isinstance(subscript_start, Node):
        if subscript_start.type == syms.listmaker:
            return False
        if subscript_start.type == syms.subscriptlist:
            subscript_start = child_towards(subscript_start, leaf)
    return subscript_start is not None and any((n.type in TEST_DESCENDANTS for n in subscript_start.pre_order()))