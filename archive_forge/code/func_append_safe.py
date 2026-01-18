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
def append_safe(self, leaf: Leaf, preformatted: bool=False) -> None:
    """Like :func:`append()` but disallow invalid standalone comment structure.

        Raises ValueError when any `leaf` is appended after a standalone comment
        or when a standalone comment is not the first leaf on the line.
        """
    if self.bracket_tracker.depth == 0 or self.bracket_tracker.any_open_for_or_lambda():
        if self.is_comment:
            raise ValueError('cannot append to standalone comments')
        if self.leaves and leaf.type == STANDALONE_COMMENT:
            raise ValueError('cannot append standalone comments to a populated line')
    self.append(leaf, preformatted=preformatted)