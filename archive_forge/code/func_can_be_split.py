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
def can_be_split(line: Line) -> bool:
    """Return False if the line cannot be split *for sure*.

    This is not an exhaustive search but a cheap heuristic that we can use to
    avoid some unfortunate formattings (mostly around wrapping unsplittable code
    in unnecessary parentheses).
    """
    leaves = line.leaves
    if len(leaves) < 2:
        return False
    if leaves[0].type == token.STRING and leaves[1].type == token.DOT:
        call_count = 0
        dot_count = 0
        next = leaves[-1]
        for leaf in leaves[-2::-1]:
            if leaf.type in OPENING_BRACKETS:
                if next.type not in CLOSING_BRACKETS:
                    return False
                call_count += 1
            elif leaf.type == token.DOT:
                dot_count += 1
            elif leaf.type == token.NAME:
                if not (next.type == token.DOT or next.type in OPENING_BRACKETS):
                    return False
            elif leaf.type not in CLOSING_BRACKETS:
                return False
            if dot_count > 1 and call_count > 1:
                return False
    return True