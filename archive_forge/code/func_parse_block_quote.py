import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_block_quote(self, m: Match, state: BlockState) -> int:
    """Parse token for block quote. Here is an example of the syntax:

        .. code-block:: markdown

            > a block quote starts
            > with right arrows
        """
    text, end_pos = self.extract_block_quote(m, state)
    child = state.child_state(text)
    if state.depth() >= self.max_nested_level - 1:
        rules = list(self.block_quote_rules)
        rules.remove('block_quote')
    else:
        rules = self.block_quote_rules
    self.parse(child, rules)
    token = {'type': 'block_quote', 'children': child.tokens}
    if end_pos:
        state.prepend_token(token)
        return end_pos
    state.append_token(token)
    return state.cursor