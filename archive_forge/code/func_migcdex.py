from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    """Compute extended gcd for multiple integers.

    Explanation
    ===========

    Given the integers $x_1, \\cdots, x_n$ and
    an extended gcd for multiple arguments are defined as a solution
    $(y_1, \\cdots, y_n), g$ for the diophantine equation
    $x_1 y_1 + \\cdots + x_n y_n = g$ such that
    $g = \\gcd(x_1, \\cdots, x_n)$.

    Examples
    ========

    >>> from sympy.functions.elementary._trigonometric_special import migcdex
    >>> migcdex()
    ((), 0)
    >>> migcdex(4)
    ((1,), 4)
    >>> migcdex(4, 6)
    ((-1, 1), 2)
    >>> migcdex(6, 10, 15)
    ((1, 1, -1), 1)
    """
    if not x:
        return ((), 0)
    if len(x) == 1:
        return ((1,), x[0])
    if len(x) == 2:
        u, v, h = igcdex(x[0], x[1])
        return ((u, v), h)
    y, g = migcdex(*x[1:])
    u, v, h = igcdex(x[0], g)
    return ((u, *(v * i for i in y)), h)