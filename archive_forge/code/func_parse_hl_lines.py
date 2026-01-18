from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
def parse_hl_lines(expr: str) -> list[int]:
    """Support our syntax for emphasizing certain lines of code.

    `expr` should be like '1 2' to emphasize lines 1 and 2 of a code block.
    Returns a list of integers, the line numbers to emphasize.
    """
    if not expr:
        return []
    try:
        return list(map(int, expr.split()))
    except ValueError:
        return []