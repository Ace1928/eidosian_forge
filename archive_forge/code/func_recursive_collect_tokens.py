from __future__ import annotations
from collections.abc import Generator, Sequence
import textwrap
from typing import Any, NamedTuple, TypeVar, overload
from .token import Token
def recursive_collect_tokens(node: _NodeType, token_list: list[Token]) -> None:
    if node.type == 'root':
        for child in node.children:
            recursive_collect_tokens(child, token_list)
    elif node.token:
        token_list.append(node.token)
    else:
        assert node.nester_tokens
        token_list.append(node.nester_tokens.opening)
        for child in node.children:
            recursive_collect_tokens(child, token_list)
        token_list.append(node.nester_tokens.closing)