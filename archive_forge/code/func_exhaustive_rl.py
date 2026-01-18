from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def exhaustive_rl(expr: _T) -> _T:
    new, old = (rule(expr), expr)
    while new != old:
        new, old = (rule(new), new)
    return new