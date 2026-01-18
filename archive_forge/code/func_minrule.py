from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def minrule(expr: _S) -> _T:
    return min([rule(expr) for rule in rules], key=objective)