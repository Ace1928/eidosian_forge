import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def extract_block_quote(self, m: Match, state: BlockState) -> Tuple[str, int]:
    """Extract text and cursor end position of a block quote."""
    text = m.group('quote_1') + '\n'
    text = expand_leading_tab(text, 3)
    text = _BLOCK_QUOTE_TRIM.sub('', text)
    sc = self.compile_sc(['blank_line', 'indent_code', 'fenced_code'])
    require_marker = bool(sc.match(text))
    state.cursor = m.end() + 1
    end_pos = None
    if require_marker:
        m = _STRICT_BLOCK_QUOTE.match(state.src, state.cursor)
        if m:
            quote = m.group(0)
            quote = _BLOCK_QUOTE_LEADING.sub('', quote)
            quote = expand_leading_tab(quote, 3)
            quote = _BLOCK_QUOTE_TRIM.sub('', quote)
            text += quote
            state.cursor = m.end()
    else:
        prev_blank_line = False
        break_sc = self.compile_sc(['blank_line', 'thematic_break', 'fenced_code', 'list', 'block_html'])
        while state.cursor < state.cursor_max:
            m = _STRICT_BLOCK_QUOTE.match(state.src, state.cursor)
            if m:
                quote = m.group(0)
                quote = _BLOCK_QUOTE_LEADING.sub('', quote)
                quote = expand_leading_tab(quote, 3)
                quote = _BLOCK_QUOTE_TRIM.sub('', quote)
                text += quote
                state.cursor = m.end()
                if not quote.strip():
                    prev_blank_line = True
                else:
                    prev_blank_line = bool(_LINE_BLANK_END.search(quote))
                continue
            if prev_blank_line:
                break
            m = break_sc.match(state.src, state.cursor)
            if m:
                end_pos = self.parse_method(m, state)
                if end_pos:
                    break
            pos = state.find_line_end()
            line = state.get_text(pos)
            line = expand_leading_tab(line, 3)
            text += line
            state.cursor = pos
    return (expand_tab(text), end_pos)