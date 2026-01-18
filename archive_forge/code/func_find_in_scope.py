from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def find_in_scope(expression, expression_types, bfs=True):
    """
    Returns the first node in this scope which matches at least one of the specified types.

    This does NOT traverse into subscopes.

    Args:
        expression (exp.Expression):
        expression_types (tuple[type]|type): the expression type(s) to match.
        bfs (bool): True to use breadth-first search, False to use depth-first.

    Returns:
        exp.Expression: the node which matches the criteria or None if no node matching
        the criteria was found.
    """
    return next(find_all_in_scope(expression, expression_types, bfs=bfs), None)