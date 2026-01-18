from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def dollar_post(src: str, end: int) -> bool:
    try:
        nxt = src[end + 1] and charCodeAt(src[end + 1], 0)
    except IndexError:
        return True
    return not nxt or nxt < 48 or nxt > 57