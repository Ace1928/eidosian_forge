from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def do_one_rl(expr: _T) -> _T:
    for rl in rules:
        result = rl(expr)
        if result != expr:
            return result
    return expr