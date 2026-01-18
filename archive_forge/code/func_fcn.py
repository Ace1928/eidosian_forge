from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def fcn(state: StateCore) -> None:
    tokens = state.tokens
    for i in range(2, len(tokens) - 1):
        if is_todo_item(tokens, i):
            todoify(tokens[i])
            tokens[i - 2].attrSet('class', 'task-list-item' + (' enabled' if not disable_checkboxes else ''))
            tokens[parent_token(tokens, i - 2)].attrSet('class', 'contains-task-list')