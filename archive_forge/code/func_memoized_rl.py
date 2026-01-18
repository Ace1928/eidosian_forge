from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def memoized_rl(expr: _S) -> _T:
    if expr in cache:
        return cache[expr]
    else:
        result = rule(expr)
        cache[expr] = result
        return result