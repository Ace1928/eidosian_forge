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
def right_hand_split(line: Line, mode: Mode, features: Collection[Feature]=(), omit: Collection[LeafID]=()) -> Iterator[Line]:
    """Split line into many lines, starting with the last matching bracket pair.

    If the split was by optional parentheses, attempt splitting without them, too.
    `omit` is a collection of closing bracket IDs that shouldn't be considered for
    this split.

    Note: running this function modifies `bracket_depth` on the leaves of `line`.
    """
    rhs_result = _first_right_hand_split(line, omit=omit)
    yield from _maybe_split_omitting_optional_parens(rhs_result, line, mode, features=features, omit=omit)