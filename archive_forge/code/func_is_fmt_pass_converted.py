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
def is_fmt_pass_converted(self, *, first_leaf_matches: Optional[Callable[[Leaf], bool]]=None) -> bool:
    """Is this line converted from fmt off/skip code?

        If first_leaf_matches is not None, it only returns True if the first
        leaf of converted code matches.
        """
    if len(self.leaves) != 1:
        return False
    leaf = self.leaves[0]
    if leaf.type != STANDALONE_COMMENT or leaf.fmt_pass_converted_first_leaf is None:
        return False
    return first_leaf_matches is None or first_leaf_matches(leaf.fmt_pass_converted_first_leaf)