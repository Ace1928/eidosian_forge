from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def make_checkbox(token: Token) -> Token:
    checkbox = Token('html_inline', '', 0)
    disabled_attr = 'disabled="disabled"' if disable_checkboxes else ''
    if token.content.startswith('[ ] '):
        checkbox.content = f'<input class="task-list-item-checkbox" {disabled_attr} type="checkbox">'
    elif token.content.startswith('[x] ') or token.content.startswith('[X] '):
        checkbox.content = f'<input class="task-list-item-checkbox" checked="checked" {disabled_attr} type="checkbox">'
    return checkbox