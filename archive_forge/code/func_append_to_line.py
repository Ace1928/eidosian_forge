import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def append_to_line(leaf: Leaf) -> Iterator[Line]:
    """Append `leaf` to current line or to new line if appending impossible."""
    nonlocal current_line
    try:
        current_line.append_safe(leaf, preformatted=True)
    except ValueError:
        yield current_line
        current_line = Line(line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
        current_line.append(leaf)