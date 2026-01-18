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
def opens_block(self) -> bool:
    """Does this line open a new level of indentation."""
    if len(self.leaves) == 0:
        return False
    return self.leaves[-1].type == token.COLON