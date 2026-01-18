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
def remove_trailing_comma(self) -> None:
    """Remove the trailing comma and moves the comments attached to it."""
    trailing_comma = self.leaves.pop()
    trailing_comma_comments = self.comments.pop(id(trailing_comma), [])
    self.comments.setdefault(id(self.leaves[-1]), []).extend(trailing_comma_comments)