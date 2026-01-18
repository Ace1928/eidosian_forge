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
def contains_standalone_comments(self) -> bool:
    """If so, needs to be split before emitting."""
    for leaf in self.leaves:
        if leaf.type == STANDALONE_COMMENT:
            return True
    return False