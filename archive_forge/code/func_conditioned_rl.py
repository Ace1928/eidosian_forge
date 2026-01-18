from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def conditioned_rl(expr: _T) -> _T:
    if cond(expr):
        return rule(expr)
    return expr