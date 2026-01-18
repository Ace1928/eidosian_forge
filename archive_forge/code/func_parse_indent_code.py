import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_indent_code(self, m: Match, state: BlockState) -> int:
    """Parse token for code block which is indented by 4 spaces."""
    end_pos = state.append_paragraph()
    if end_pos:
        return end_pos
    code = m.group(0)
    code = expand_leading_tab(code)
    code = _INDENT_CODE_TRIM.sub('', code)
    code = code.strip('\n')
    state.append_token({'type': 'block_code', 'raw': code, 'style': 'indent'})
    return m.end()