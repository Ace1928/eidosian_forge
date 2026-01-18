from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def dollar_pre(src: str, beg: int) -> bool:
    prv = charCodeAt(src[beg - 1], 0) if beg > 0 else False
    return not prv or (prv != 92 and (prv < 48 or prv > 57))