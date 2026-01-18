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
def has_magic_trailing_comma(self, closing: Leaf) -> bool:
    """Return True if we have a magic trailing comma, that is when:
        - there's a trailing comma here
        - it's not from single-element square bracket indexing
        - it's not a one-tuple
        """
    if not (closing.type in CLOSING_BRACKETS and self.leaves and (self.leaves[-1].type == token.COMMA)):
        return False
    if closing.type == token.RBRACE:
        return True
    if closing.type == token.RSQB:
        if closing.parent is not None and closing.parent.type == syms.trailer and (closing.opening_bracket is not None) and is_one_sequence_between(closing.opening_bracket, closing, self.leaves, brackets=(token.LSQB, token.RSQB)):
            assert closing.prev_sibling is not None
            assert closing.prev_sibling.type == syms.subscriptlist
            return False
        return True
    if self.is_import:
        return True
    if closing.opening_bracket is not None and (not is_one_sequence_between(closing.opening_bracket, closing, self.leaves)):
        return True
    return False