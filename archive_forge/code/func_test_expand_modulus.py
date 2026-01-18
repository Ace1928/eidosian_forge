from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import expand, expand_multinomial, expand_power_base
from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically
from sympy.abc import x, y, z
def test_expand_modulus():
    assert ((x + y) ** 11).expand(modulus=11) == x ** 11 + y ** 11
    assert ((x + sqrt(2) * y) ** 11).expand(modulus=11) == x ** 11 + 10 * sqrt(2) * y ** 11
    assert (x + y / 2).expand(modulus=1) == y / 2
    raises(ValueError, lambda: ((x + y) ** 11).expand(modulus=0))
    raises(ValueError, lambda: ((x + y) ** 11).expand(modulus=x))