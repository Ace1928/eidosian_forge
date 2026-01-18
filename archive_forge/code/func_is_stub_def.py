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
def is_stub_def(self) -> bool:
    """Is this line a function definition with a body consisting only of "..."?"""
    return self.is_def and self.leaves[-4:] == [Leaf(token.COLON, ':')] + [Leaf(token.DOT, '.') for _ in range(3)]