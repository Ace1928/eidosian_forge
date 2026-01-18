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
def bench_R7():
    """[f.subs(x, random()) for _ in range(10**4)]"""
    f = x ** 24 + 34 * x ** 12 + 45 * x ** 3 + 9 * x ** 18 + 34 * x ** 10 + 32 * x ** 21
    [f.subs(x, random()) for _ in range(10 ** 4)]