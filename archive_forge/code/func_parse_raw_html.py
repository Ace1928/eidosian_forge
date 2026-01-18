import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_raw_html(self, m: Match, state: BlockState) -> Optional[int]:
    marker = m.group(0).strip()
    if marker == '<!--':
        return _parse_html_to_end(state, '-->', m.end())
    if marker == '<?':
        return _parse_html_to_end(state, '?>', m.end())
    if marker == '<![CDATA[':
        return _parse_html_to_end(state, ']]>', m.end())
    if marker.startswith('<!'):
        return _parse_html_to_end(state, '>', m.end())
    close_tag = None
    open_tag = None
    if marker.startswith('</'):
        close_tag = marker[2:].lower()
        if close_tag in BLOCK_TAGS:
            return _parse_html_to_newline(state, self.BLANK_LINE)
    else:
        open_tag = marker[1:].lower()
        if open_tag in PRE_TAGS:
            end_tag = '</' + open_tag + '>'
            return _parse_html_to_end(state, end_tag, m.end())
        if open_tag in BLOCK_TAGS:
            return _parse_html_to_newline(state, self.BLANK_LINE)
    end_pos = state.append_paragraph()
    if end_pos:
        return end_pos
    start_pos = m.end()
    end_pos = state.find_line_end()
    if open_tag and _OPEN_TAG_END.match(state.src, start_pos, end_pos) or (close_tag and _CLOSE_TAG_END.match(state.src, start_pos, end_pos)):
        return _parse_html_to_newline(state, self.BLANK_LINE)