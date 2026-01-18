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
def should_split_funcdef_with_rhs(line: Line, mode: Mode) -> bool:
    """If a funcdef has a magic trailing comma in the return type, then we should first
    split the line with rhs to respect the comma.
    """
    return_type_leaves: List[Leaf] = []
    in_return_type = False
    for leaf in line.leaves:
        if leaf.type == token.COLON:
            in_return_type = False
        if in_return_type:
            return_type_leaves.append(leaf)
        if leaf.type == token.RARROW:
            in_return_type = True
    result = Line(mode=line.mode, depth=line.depth)
    leaves_to_track = get_leaves_inside_matching_brackets(return_type_leaves)
    for leaf in return_type_leaves:
        result.append(leaf, preformatted=True, track_bracket=id(leaf) in leaves_to_track)
    return result.magic_trailing_comma is not None