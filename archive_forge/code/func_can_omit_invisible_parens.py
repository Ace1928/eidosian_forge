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
def can_omit_invisible_parens(rhs: RHSResult, line_length: int) -> bool:
    """Does `rhs.body` have a shape safe to reformat without optional parens around it?

    Returns True for only a subset of potentially nice looking formattings but
    the point is to not return false positives that end up producing lines that
    are too long.
    """
    line = rhs.body
    closing_bracket: Optional[Leaf] = None
    for leaf in reversed(line.leaves):
        if closing_bracket and leaf is closing_bracket.opening_bracket:
            closing_bracket = None
        if leaf.type == STANDALONE_COMMENT and (not closing_bracket):
            return False
        if not closing_bracket and leaf.type in CLOSING_BRACKETS and (leaf.opening_bracket in line.leaves) and leaf.value:
            closing_bracket = leaf
    bt = line.bracket_tracker
    if not bt.delimiters:
        return True
    max_priority = bt.max_delimiter_priority()
    delimiter_count = bt.delimiter_count_with_priority(max_priority)
    if delimiter_count > 1:
        return False
    if delimiter_count == 1:
        if max_priority == COMMA_PRIORITY and rhs.head.is_with_or_async_with_stmt:
            return False
    if max_priority == DOT_PRIORITY:
        return True
    assert len(line.leaves) >= 2, 'Stranded delimiter'
    first = line.leaves[0]
    second = line.leaves[1]
    if first.type in OPENING_BRACKETS and second.type not in CLOSING_BRACKETS:
        if _can_omit_opening_paren(line, first=first, line_length=line_length):
            return True
    penultimate = line.leaves[-2]
    last = line.leaves[-1]
    if last.type == token.RPAR or last.type == token.RBRACE or (last.type == token.RSQB and last.parent and (last.parent.type != syms.trailer)):
        if penultimate.type in OPENING_BRACKETS:
            return False
        if is_multiline_string(first):
            return True
        if _can_omit_closing_paren(line, last=last, line_length=line_length):
            return True
    return False