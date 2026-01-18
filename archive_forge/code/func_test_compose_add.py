from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_compose_add():
    R, x = ring('x', QQ)
    p1 = x ** 3 - 1
    p2 = x ** 2 - 2
    assert rs_compose_add(p1, p2) == x ** 6 - 6 * x ** 4 - 2 * x ** 3 + 12 * x ** 2 - 12 * x - 7