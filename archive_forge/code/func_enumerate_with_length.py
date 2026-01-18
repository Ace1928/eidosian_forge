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
def enumerate_with_length(self, is_reversed: bool=False) -> Iterator[Tuple[Index, Leaf, int]]:
    """Return an enumeration of leaves with their length.

        Stops prematurely on multiline strings and standalone comments.
        """
    op = cast(Callable[[Sequence[Leaf]], Iterator[Tuple[Index, Leaf]]], enumerate_reversed if is_reversed else enumerate)
    for index, leaf in op(self.leaves):
        length = len(leaf.prefix) + len(leaf.value)
        if '\n' in leaf.value:
            return
        for comment in self.comments_after(leaf):
            length += len(comment.value)
        yield (index, leaf, length)