from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def find_all_in_scope(expression, expression_types, bfs=True):
    """
    Returns a generator object which visits all nodes in this scope and only yields those that
    match at least one of the specified expression types.

    This does NOT traverse into subscopes.

    Args:
        expression (exp.Expression):
        expression_types (tuple[type]|type): the expression type(s) to match.
        bfs (bool): True to use breadth-first search, False to use depth-first.

    Yields:
        exp.Expression: nodes
    """
    for expression in walk_in_scope(expression, bfs=bfs):
        if isinstance(expression, tuple(ensure_collection(expression_types))):
            yield expression