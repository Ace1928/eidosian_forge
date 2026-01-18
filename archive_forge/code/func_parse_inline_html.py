import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def parse_inline_html(self, m: Match, state: InlineState) -> int:
    end_pos = m.end()
    html = m.group(0)
    state.append_token({'type': 'inline_html', 'raw': html})
    if html.startswith(('<a ', '<a>', '<A ', '<A>')):
        state.in_link = True
    elif html.startswith(('</a ', '</a>', '</A ', '</A>')):
        state.in_link = False
    return end_pos