from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def is_todo_item(tokens: list[Token], index: int) -> bool:
    return is_inline(tokens[index]) and is_paragraph(tokens[index - 1]) and is_list_item(tokens[index - 2]) and starts_with_todo_markdown(tokens[index])