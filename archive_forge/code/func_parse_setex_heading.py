import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_setex_heading(self, m: Match, state: BlockState) -> Optional[int]:
    """Parse token for setex style heading. A setex heading syntax looks like:

        .. code-block:: markdown

            H1 title
            ========
        """
    last_token = state.last_token()
    if last_token and last_token['type'] == 'paragraph':
        level = 1 if m.group('setext_1') == '=' else 2
        last_token['type'] = 'heading'
        last_token['style'] = 'setext'
        last_token['attrs'] = {'level': level}
        return m.end() + 1
    sc = self.compile_sc(['thematic_break', 'list'])
    m = sc.match(state.src, state.cursor)
    if m:
        return self.parse_method(m, state)