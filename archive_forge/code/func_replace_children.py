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
def replace_children(expression: Expression, fun: t.Callable, *args, **kwargs) -> None:
    """
    Replace children of an expression with the result of a lambda fun(child) -> exp.
    """
    for k, v in tuple(expression.args.items()):
        is_list_arg = type(v) is list
        child_nodes = v if is_list_arg else [v]
        new_child_nodes = []
        for cn in child_nodes:
            if isinstance(cn, Expression):
                for child_node in ensure_collection(fun(cn, *args, **kwargs)):
                    new_child_nodes.append(child_node)
            else:
                new_child_nodes.append(cn)
        expression.set(k, new_child_nodes if is_list_arg else seq_get(new_child_nodes, 0))