from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_cos_sin():
    R, x, y = ring('x, y', QQ)
    cos, sin = rs_cos_sin(x, x, 9)
    assert cos == rs_cos(x, x, 9)
    assert sin == rs_sin(x, x, 9)
    cos, sin = rs_cos_sin(x + x * y, x, 5)
    assert cos == rs_cos(x + x * y, x, 5)
    assert sin == rs_sin(x + x * y, x, 5)