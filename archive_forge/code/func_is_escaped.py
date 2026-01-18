from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, isWhiteSpace
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def is_escaped(state: StateInline, back_pos: int, mod: int=0) -> bool:
    """Test if dollar is escaped."""
    backslashes = 0
    while back_pos >= 0:
        back_pos = back_pos - 1
        if state.src[back_pos] == '\\':
            backslashes += 1
        else:
            break
    if not backslashes:
        return False
    if backslashes % 2 != mod:
        return True
    return False