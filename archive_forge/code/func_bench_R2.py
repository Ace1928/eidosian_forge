from sympy.core.random import random
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import factor
from sympy.simplify.simplify import simplify
from sympy.abc import x, y, z
from timeit import default_timer as clock
def bench_R2():
    """Hermite polynomial hermite(15, y)"""

    def hermite(n, y):
        if n == 1:
            return 2 * y
        if n == 0:
            return 1
        return (2 * y * hermite(n - 1, y) - 2 * (n - 1) * hermite(n - 2, y)).expand()
    hermite(15, y)