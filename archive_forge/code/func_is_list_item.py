from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def is_list_item(token: Token) -> bool:
    return token.type == 'list_item_open'