from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_mul_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = 1 + t * x + t * y
    for i in range(2):
        p = rs_mul(p, p, t, 3)
    assert p == 6 * x ** 2 * t ** 2 + 12 * x * y * t ** 2 + 6 * y ** 2 * t ** 2 + 4 * x * t + 4 * y * t + 1
    p = 1 + t * x + t * y + t ** 2 * x * y
    p1 = rs_mul(p, p, t, 2)
    assert p1 == 1 + 2 * t * x + 2 * t * y
    R1, z = ring('z', QQ)
    raises(ValueError, lambda: rs_mul(p, z, x, 2))
    p1 = 2 + 2 * x + 3 * x ** 2
    p2 = 3 + x ** 2
    assert rs_mul(p1, p2, x, 4) == 2 * x ** 3 + 11 * x ** 2 + 6 * x + 6