from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def rl2(x: int) -> int:
    if x == 2:
        return 3
    return x