import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def describe_token_expr(expr: str) -> str:
    """Like `describe_token` but for token expressions."""
    if ':' in expr:
        type, value = expr.split(':', 1)
        if type == TOKEN_NAME:
            return value
    else:
        type = expr
    return _describe_token_type(type)