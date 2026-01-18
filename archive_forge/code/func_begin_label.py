from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def begin_label() -> Token:
    token = Token('html_inline', '', 0)
    token.content = '<label>'
    return token