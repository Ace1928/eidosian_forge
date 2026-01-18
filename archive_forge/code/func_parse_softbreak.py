import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def parse_softbreak(self, m: Match, state: InlineState) -> int:
    state.append_token({'type': 'softbreak'})
    return m.end()