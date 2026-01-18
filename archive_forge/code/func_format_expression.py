from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
@staticmethod
def format_expression(left: 'RedisFilterExpression', right: 'RedisFilterExpression', operator_str: str) -> str:
    _left, _right = (str(left), str(right))
    if _left == _right == '*':
        return _left
    if _left == '*' != _right:
        return _right
    if _right == '*' != _left:
        return _left
    return f'({_left}{operator_str}{_right})'