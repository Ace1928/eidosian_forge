from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def replace_tree(expression: Expression, fun: t.Callable, prune: t.Optional[t.Callable[[Expression], bool]]=None) -> Expression:
    """
    Replace an entire tree with the result of function calls on each node.

    This will be traversed in reverse dfs, so leaves first.
    If new nodes are created as a result of function calls, they will also be traversed.
    """
    stack = list(expression.dfs(prune=prune))
    while stack:
        node = stack.pop()
        new_node = fun(node)
        if new_node is not node:
            node.replace(new_node)
            if isinstance(new_node, Expression):
                stack.append(new_node)
    return new_node