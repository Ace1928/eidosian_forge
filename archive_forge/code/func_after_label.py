from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def after_label(content: str, checkbox_id: str) -> Token:
    token = Token('html_inline', '', 0)
    token.content = f'<label class="task-list-item-label" for="{checkbox_id}">{content}</label>'
    token.attrs = {'for': checkbox_id}
    return token