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
def generate_trailers_to_omit(line: Line, line_length: int) -> Iterator[Set[LeafID]]:
    """Generate sets of closing bracket IDs that should be omitted in a RHS.

    Brackets can be omitted if the entire trailer up to and including
    a preceding closing bracket fits in one line.

    Yielded sets are cumulative (contain results of previous yields, too).  First
    set is empty, unless the line should explode, in which case bracket pairs until
    the one that needs to explode are omitted.
    """
    omit: Set[LeafID] = set()
    if not line.magic_trailing_comma:
        yield omit
    length = 4 * line.depth
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    inner_brackets: Set[LeafID] = set()
    for index, leaf, leaf_length in line.enumerate_with_length(is_reversed=True):
        length += leaf_length
        if length > line_length:
            break
        has_inline_comment = leaf_length > len(leaf.value) + len(leaf.prefix)
        if leaf.type == STANDALONE_COMMENT or has_inline_comment:
            break
        if opening_bracket:
            if leaf is opening_bracket:
                opening_bracket = None
            elif leaf.type in CLOSING_BRACKETS:
                prev = line.leaves[index - 1] if index > 0 else None
                if prev and prev.type == token.COMMA and (leaf.opening_bracket is not None) and (not is_one_sequence_between(leaf.opening_bracket, leaf, line.leaves)):
                    break
                inner_brackets.add(id(leaf))
        elif leaf.type in CLOSING_BRACKETS:
            prev = line.leaves[index - 1] if index > 0 else None
            if prev and prev.type in OPENING_BRACKETS:
                inner_brackets.add(id(leaf))
                continue
            if closing_bracket:
                omit.add(id(closing_bracket))
                omit.update(inner_brackets)
                inner_brackets.clear()
                yield omit
            if prev and prev.type == token.COMMA and (leaf.opening_bracket is not None) and (not is_one_sequence_between(leaf.opening_bracket, leaf, line.leaves)):
                break
            if leaf.value:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf