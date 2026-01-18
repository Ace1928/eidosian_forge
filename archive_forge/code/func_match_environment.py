from __future__ import annotations
import re
from typing import TYPE_CHECKING, Callable, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def match_environment(string: str) -> None | tuple[str, str, int]:
    match_open = RE_OPEN.match(string)
    if not match_open:
        return None
    environment = match_open.group(1)
    numbered = match_open.group(2)
    match_close = re.search('\\\\end\\{' + environment + numbered.replace('*', '\\*') + '\\}', string)
    if not match_close:
        return None
    return (environment, numbered, match_close.end())