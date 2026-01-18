import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_block_html(self, m: Match, state: BlockState) -> Optional[int]:
    return self.parse_raw_html(m, state)