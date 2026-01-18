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
def all_lines(self) -> List[str]:
    empty_line = str(Line(mode=self.mode))
    prefix = make_simple_prefix(self.before, self.form_feed, empty_line)
    return [prefix] + self.content_lines + [empty_line * self.after]