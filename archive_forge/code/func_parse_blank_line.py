import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_blank_line(self, m: Match, state: BlockState) -> int:
    """Parse token for blank lines."""
    state.append_token({'type': 'blank_line'})
    return m.end()