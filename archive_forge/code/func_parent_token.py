from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def parent_token(tokens: list[Token], index: int) -> int:
    target_level = tokens[index].level - 1
    for i in range(1, index + 1):
        if tokens[index - i].level == target_level:
            return index - i
    return -1